from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
from PIL import Image, ImageDraw
import yaml

# whether work as a module
try:
    from utils import get_file_list, load_annt_file
except ImportError:
    from .utils import get_file_list, load_annt_file

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class refine_db(object):
	def __init__(self, cfg_path, db_name, cache_dir='data/cache', save_patch=None):

		with open(cfg_path, 'r') as f:
			data_cfg = yaml.load(f)
		f.close()
		print(cfg_path)

		self._img_dir = data_cfg['img_dir']
		self._gt_dir = data_cfg['gt_dir']
		self._data_dir = data_cfg['anchor_dir']
		self._db_name = data_cfg['db_name']
		self._suffix = data_cfg['image_ext']
		self._classes = data_cfg['classes']

		self._cache_dir = cache_dir
		self._num_classes = len(self._classes)
		self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
		self._save_patch = save_patch

		self._refinedb = self._refine_database()

	@property
	def refinedb(self):
		return self._refinedb

	# generate _refinedb to refine detection result
	def _refine_database(self):
		# load cached data if it existed.
		cache_file = os.path.join(self._cache_dir, self._db_name+'_gt_db.pkl')
		if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				refinedb = pickle.load(fid)
			print('{} database loaded from {}'.format(self._db_name, cache_file))
			return refinedb

        # generate db used for refine result.
        # file list is generated from predict xml files
		file_name_itr = get_file_list(self._data_dir, file_suffix='.xml')
		refine_boxes = []
		# refine_boxes_ext = []
		refine_class = []
		refine_img = []
		img_path_list = []
		for file_name in file_name_itr:
			print(file_name)
			annt_file = os.path.join(self._data_dir, file_name+'.xml')
			gt_annt_file = os.path.join(self._gt_dir, file_name+'.xml')
			img_file = os.path.join(self._img_dir, file_name+self._suffix)
			img = Image.open(img_file)
			try:
				# boxes, boxes_ext, classes, patches = self._get_box_info(annt_file, gt_annt_file, img)
				boxes, classes, patches = self._get_box_info(annt_file, gt_annt_file, img)
			except ValueError as e:
				print(e)
				print(annt_file)
				print(gt_annt_file)
				print(img_file)
				classes = []

			if len(classes)==0:
				continue
			img_path_list.extend([os.path.join(self._img_dir, file_name+self._suffix)]*len(boxes)) 
			refine_boxes.extend(boxes)
			# refine_boxes_ext.extend(boxes_ext)
			refine_class.extend(classes)
			refine_img.extend(patches)

		# db_info = zip(refine_boxes, refine_boxes_ext, refine_class, img_path_list, refine_img)
		# refinedb = [dict(zip(('box', 'box_ext', 'class', 'path', 'image'), item_info)) for item_info in db_info]
		db_info = zip(refine_boxes, refine_class, img_path_list, refine_img)
		refinedb = [dict(zip(('box', 'class', 'path', 'image'), item_info)) for item_info in db_info]
		with open(cache_file, 'wb') as fid:
			pickle.dump(refinedb, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote {} database to {}'.format(self._db_name, cache_file))
		return refinedb

    # generate refined info for an xml file
	def _get_box_info(self, annt_file, gt_annt_file, img):
		if not(os.path.exists(gt_annt_file)):
			print(''''{} doesn\'t exist. All the boxes will be recognized as background.'''.format(gt_annt_file))
			gt_info = None
		else:
			gt_info, elemet_tree = load_annt_file(gt_annt_file)
		annt_info, elemet_tree = load_annt_file(annt_file)
		boxes = []
		# boxes_ext = []
		classes = []
		patch = []
		if len(annt_info['classes_name']) == 0:
			# return boxes, boxes_ext, classes, patch
			return boxes, classes, patch

		for box in annt_info['boxes']:
			box_class, usage_flag = self._get_box_class(box, gt_info)
			if usage_flag:
				box_ext = self._expand_box(box, img.size)
				# boxes_ext.append(box_ext.tolist())
				boxes.append(box.tolist())
				classes.append(box_class)
				if self._save_patch:
					patch.append(np.asarray(img.crop(box_ext.tolist())))
				else:
					patch.append(np.asarray(img.crop(box.tolist())))

		# return boxes, boxes_ext, classes, patch
		return boxes, classes, patch

	def _get_box_class(self, box, gt_info=None):
		if gt_info==None:
			return 0, True
		gt_boxes = gt_info['boxes']
		gt_classes = gt_info['classes_name']
		overlap_list = [self._get_overlap(box, gt_box) for gt_box in gt_boxes]
		overlap_array = np.array(overlap_list)
		if len(overlap_array) == 0:
			return 0, True
		if overlap_array.max() <0.1:
			return 0, True
		elif overlap_array.max() >0.5:
			index_max = np.where(abs(overlap_array-overlap_array.max())<1e-10)
			current_classname = gt_classes[index_max[0][0]]
			if not(current_classname in self._classes):
				if current_classname=='hsil':
					current_classname = "HSIL"
				else:
					current_classname = "abnormal"
			return self._class_to_ind[current_classname], True
		else:
			return 0, False

	def _get_overlap(self, box, gt_box):
		xmin_box, ymin_box, xmax_box, ymax_box = box
		xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_box
		xmin = max(xmin_box, xmin_gt)
		ymin = max(ymin_box, ymin_gt)
		xmax = min(xmax_box, xmax_gt)
		ymax = min(ymax_box, ymax_gt)
		w = float(xmax) - float(xmin)
		h = float(ymax) - float(ymin)
		if w <=0 or h <=0:
			return 0
		else:
			return w*h/(float(xmax_box-xmin_box)*float(ymax_box-ymin_box))



	# # save patches
	def _expand_box(self, box, img_size):
		l_x = (box[2] - box[0]) * 2
		l_y = (box[3] - box[1]) * 2
		x1 = 0 if int(box[0])-l_x<0 else box[0]-l_x
		y1 = 0 if int(box[1])-l_y<0 else box[1]-l_y
		x2 = img_size[0]-1 if box[2]+l_x>img_size[0]-1 else box[2]+l_x
		y2 = img_size[1]-1 if box[3]+l_y>img_size[1]-1 else box[3]+l_y
		box = np.array([x1, y1, x2, y2])
		return box

	# def _plot_bndbox(self, box, box_ext, img):
	# 	xmin = box[0] - box_ext[0]
	# 	ymin = box[1] - box_ext[1]
	# 	xmax = box[2] - box_ext[0]
	# 	ymax = box[3] - box_ext[1]

	# 	draw = ImageDraw.Draw(img)
	# 	draw.rectangle([xmin,ymin,xmax,ymax], outline=(0,255,0))
	# 	draw.rectangle([xmin+1,ymin+1,xmax-1,ymax-1], outline=(0,255,0))
	# 	draw.rectangle([xmin+2,ymin+2,xmax-2,ymax-2], outline=(0,255,0))

	# 	return img


	# def _save_bndboxes(self, save_dir):
	# 	for index, item in enumerate(self._refinedb):
	# 		patch_img = Image.fromarray(item['image'])

	# 		if self._save_patch:
	# 			patch_img = self._plot_bndbox(item['box'], item['box_ext'], patch_img)
				
	# 		img_path_save = os.path.join(save_dir, '%05d.bmp' % index)
	# 		patch_img.save(img_path_save)



if __name__ == '__main__':
	# # trainning set
	# img_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/train_raw'
	# gt_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/train_raw'
	# data_dir = '/home/lc/code/faster-rcnn.pytorch/output/restore/train'

	# db = refine_db(img_dir, gt_dir, data_dir, 'refine', cache_dir='/home/lc/code/NetKit/data/cache', img_suffix='.jpg')
	# test = db.refinedb

	# save_dir = '/home/lc/code/NetKit/data/output/patch_to_review/train'
	# db.save_bndboxes(save_dir)

	# # test set
	# img_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/test_raw'
	# gt_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/test_raw'
	# data_dir = '/home/lc/code/faster-rcnn.pytorch/output/restore/test'

	# db = refine_db(img_dir, gt_dir, data_dir, 'refine_test', cache_dir='/home/lc/code/NetKit/data/cache', img_suffix='.jpg')
	# test = db.refinedb

	# save_dir = '/home/lc/code/NetKit/data/output/patch_to_review/test'
	# db.save_bndboxes(save_dir)

	# all negtive data
	# img_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/all_neg/train_raw'
	# gt_dir = ''
	# data_dir = '/home/lc/code/faster-rcnn.pytorch/output/restore/all_neg'

	# db = refine_db(img_dir, gt_dir, data_dir, 'neg', cache_dir='/home/lc/code/NetKit/data/cache', img_suffix='.jpg')
	# test = db.refinedb

	######################################################################################
	# save patch
	# img_dir = '/home/lc/code/NetKit/data/own_data/patch_extract_0723'
	# gt_dir = '/home/lc/code/NetKit/data/own_data/patch_extract_0723'
	# data_dir = '/home/lc/code/NetKit/data/own_data/patch_extract_0723'

	# db = refine_db(img_dir, gt_dir, data_dir, 'refine0723', cache_dir='/home/lc/code/NetKit/data/cache', img_suffix='.jpg', save_patch=True)
	# test = db.refinedb

	# save_dir = '/home/lc/code/NetKit/data/output/patch_0723'
	# db._save_bndboxes(save_dir)



	# img_dir = '/home/liuchang/code/classification_net_pytorch/data/own_data/test'
	# data_dir = '/home/liuchang/code/classification_net_pytorch/data/own_data/output/cls_output/test'
	# gt_dir = '/home/liuchang/code/classification_net_pytorch/data/own_data/test'
	img_dir = '/home/zhouming/data/cc/train'
	data_dir = '/home/zhouming/data/cc/output/annt_info_outdir20'
	gt_dir = '/home/zhouming/data/cc/train'
	cfg_path = '/home/zhouming/code/CCBackEnd/clsNet/cfgs/data_cfg_refine_db.yml'

	#db = refine_db(img_dir, gt_dir, data_dir, 'test_precision_pre', cache_dir='data/cache', img_suffix='.jpg')
	db = refine_db(cfg_path = cfg_path, db_name = 'refine_db', cache_dir='/home/zhouming/code/CCBackEnd/clsNet/data/cache')
	test = db.refinedb

	class_list = [item['class'] for item in test]
	print(len(class_list))
	print(sum(class_list))

	# save_dir = '/home/lc/code/NetKit/data/output/patch_0723'
	# db._save_bndboxes(save_dir)
