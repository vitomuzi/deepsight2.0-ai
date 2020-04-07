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


class normal_cell_db(object):
	def __init__(self, cfg_path, cache_dir='data/cache'):

		with open(cfg_path, 'r') as f:
			data_cfg = yaml.load(f)
		f.close()
		print(cfg_path)

		self._img_dir = data_cfg['data_path']
		self._gt_dir = data_cfg['data_path']
		self._data_dir = data_cfg['anchor_path']
		self._db_name = data_cfg['db_name']
		self._suffix = data_cfg['image_ext']
		self._classes = data_cfg['classes']

		self._cache_dir = cache_dir
		self._num_classes = len(self._classes)
		self._class_to_ind = data_cfg['class_to_ind']

		self._normaldb = self._normal_cell_database()

	@property
	def normaldb(self):
		return self._normaldb

	# generate _refinedb to refine detection result
	def _normal_cell_database(self):
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
				boxes, classes, patches = self._get_box_info(annt_file, gt_annt_file, img)
			except ValueError as e:
				print(e)

				classes = []

			if len(classes)==0:
				continue

			img_path_list.extend([os.path.join(self._img_dir, file_name+self._suffix)]*len(boxes)) 
			refine_boxes.extend(boxes)
			refine_class.extend(classes)
			refine_img.extend(patches)

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
		classes = []
		patch = []
		if len(annt_info['classes_name']) == 0:
			return boxes, classes, patch

		for idx, box in enumerate(annt_info['boxes']):
			# only process bndboxes of label '2'
			if annt_info['classes_name'][idx] != '2':
				continue

			box_class, usage_flag = self._get_box_class(box, gt_info)
			if usage_flag:
				boxes.append(box.tolist())
				classes.append(box_class)
				patch.append(np.asarray(img.crop(box.tolist())))

		# return boxes, boxes_ext, classes, patch
		return boxes, classes, patch

	def _get_box_class(self, box, gt_info=None):
		# return class index 0 as the normal cell
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
			return self._class_to_ind[gt_classes[index_max[0][0]]], True
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


if __name__ == '__main__':

	gt_dir = '/home/liuchang/code/classification_net_pytorch/data/own_data/test'

	db = refine_db(img_dir, gt_dir, data_dir, 'test_precision_pre', cache_dir='/home/liuchang/code/classification_net_pytorch/data/cache', img_suffix='.jpg')
	test = db.refinedb

	class_list = [item['class'] for item in test]
	print(len(class_list))
	print(sum(class_list))

	# save_dir = '/home/lc/code/NetKit/data/output/patch_0723'
	# db._save_bndboxes(save_dir)
