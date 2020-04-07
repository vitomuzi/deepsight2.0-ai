import numpy as np
import math
import os
from PIL import Image
from scipy.misc import imread
import torch
import random

import general as ut

# crop large original image into some smaller images with a n*m grid. corresponding annotation files are also generated.
def _aug_img_2d(img_path, annt_path, out_dir, own_data_classes, aug_factor=60, trim_size=1024, random_seed=False, overlap_th=0.4, ratio_th=2.5):
	get_sub_path = lambda sub_index, ori_path, out_dir:os.path.join(out_dir, os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1])

	try:
		im = imread(img_path)
		annt_info, template_tree = ut.load_annt_file(annt_path)
	except ValueError as e:
		print(e)
		return
	except OSError as e:
		print(e)
		return

	if len(annt_info['gt_classes_name'])==0:
		print('no bndboxes in this file.')
		return

	# random crop
	if random_seed:
		random.seed(random_seed)
	gt_boxes_ori = torch.from_numpy(annt_info['boxes'].astype(np.float))
	class_to_ind_dict = dict(zip(own_data_classes, range(len(own_data_classes))))
	ind_to_class_dict = dict(zip(range(len(own_data_classes)), own_data_classes))
	gt_classes_np = np.array([class_to_ind_dict[item] for item in annt_info['gt_classes_name']])
	gt_classes_ori = torch.from_numpy(gt_classes_np)

	gt_area = (gt_boxes_ori[:,2]-gt_boxes_ori[:,0])*(gt_boxes_ori[:,3]-gt_boxes_ori[:,1])
	gt_ratio = (gt_boxes_ori[:,2]-gt_boxes_ori[:,0])/(gt_boxes_ori[:,3]-gt_boxes_ori[:,1])
	gt_ratio[gt_ratio>ratio_th-1e-5]=0
	gt_ratio[gt_ratio<1.0/ratio_th+1e-5]=0

	if torch.sum(gt_ratio) < 1e-5:
		return

	# crop img
	height, width = im.shape[0], im.shape[1]
	min_x_bnd = max(0, torch.min(gt_boxes_ori[:,0])-trim_size)
	min_y_bnd = max(0, torch.min(gt_boxes_ori[:,1])-trim_size)
	max_x_bnd = min(width-trim_size, torch.max(gt_boxes_ori[:,2]))
	max_y_bnd = min(height-trim_size, torch.max(gt_boxes_ori[:,3]))

	sub_index = 0
	for sub_index_x in range(aug_factor):
		sub_img_path = get_sub_path(sub_index, img_path, out_dir)
		sub_annt_path = get_sub_path(sub_index, annt_path, out_dir)

		while True:
			gt_boxes = gt_boxes_ori.clone()
			gt_classes = gt_classes_ori.clone()
			x_min_img, y_min_img = random.randint(min_x_bnd, max_x_bnd), random.randint(min_y_bnd, max_y_bnd)
			gt_boxes[:,::2] = gt_boxes[:,::2] - x_min_img
			gt_boxes[:,1::2] = gt_boxes[:,1::2] - y_min_img
			gt_boxes.clamp_(0, trim_size - 1)
			overlap = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])/gt_area
			ratio = (gt_boxes[:,2]-gt_boxes[:,0])/(gt_boxes[:,3]-gt_boxes[:,1])
			boxes_width = gt_boxes[:,2]-gt_boxes[:,0]
			boxes_height = gt_boxes[:,3]-gt_boxes[:,1]
			# remove bndbox
			not_keep = (overlap < overlap_th) | (ratio > ratio_th) | (ratio < (1/ratio_th)) | (boxes_width<5) | (boxes_height<5)
			keep = torch.nonzero(not_keep == 0).view(-1)
			if keep.numel() == 0:
				continue
		
			gt_boxes = gt_boxes[keep]
			gt_classes = gt_classes[keep]
			annt_info['boxes'] = gt_boxes.numpy().astype(np.uint16)
			annt_info['gt_classes_name'] = [ind_to_class_dict[item] for item in list(gt_classes.numpy().astype(np.int32))]
			annt_info['ishards'] = np.zeros(len(annt_info['gt_classes_name'])).astype(np.int32)
	
			img = im[
				y_min_img:y_min_img+trim_size, 
				x_min_img:x_min_img+trim_size,
				:]
			break

		pil_img = Image.fromarray(img)

		ut.save_annt_file(sub_annt_path, annt_info, template_tree)
		pil_img.save(sub_img_path)
		sub_index += 1


def _random_crop(img_path, sub_img_path,trim_size,random_seed=False):
	get_sub_path = lambda sub_index, ori_path, out_dir:os.path.join(out_dir, os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1])
	
	try:
		im = imread(img_path)
	except ValueError as e:
		print(e)
		return
	except OSError as e:
		print(e)
		return
	# random crop
	if random_seed:
		random.seed(random_seed)

	# crop img
	height, width = im.shape[0], im.shape[1]
	min_x_bnd = 0
	min_y_bnd = 0
	max_x_bnd = width-trim_size
	max_y_bnd = height-trim_size
	x_min_img, y_min_img = random.randint(min_x_bnd, max_x_bnd), random.randint(min_y_bnd, max_y_bnd)

	sub_img_path = get_sub_path(0, img_path, out_dir)

	img = im[
		y_min_img:y_min_img+trim_size, 
		x_min_img:x_min_img+trim_size,
		:]

	pil_img = Image.fromarray(img)
	pil_img.save(sub_img_path)




if __name__ == '__main__':
	# for faster rcnn================
	# aug factor:
	# neg all cell: 60, cancer all cell:25

	_dir = '/home/zhouming/data/cc/cell_det/ensemble_annot'
	#_dir = '/home/zhouming/data/cc/cell_det/neg_train'
	annt_dir = _dir
	img_dir = _dir
	#out_dir = '/home/zhouming/data/cc/cell_det/infla/ensemble_annot'
	#out_dir = '/home/zhouming/data/cc/cell_det/infla/neg_train'
	out_dir = '/home/zhouming/data/cc/cell_det/infla/ensemble_annot_2048'
	file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')

	own_data_classes=('__background__', 'abnormal')
	# own_data_classes=('__background__', '2', '3')
	
	for file_name in file_itr:
		annt_path = annt_dir+'/'+file_name+'.xml'
		img_path = img_dir+'/'+file_name+'.jpg'
		print(img_path)
		try:
			_aug_img_2d(img_path, annt_path, out_dir, own_data_classes, trim_size=2048, aug_factor=5, overlap_th=0.4, ratio_th=4)
			#_random_crop(img_path, out_dir, trim_size=2048)
		except Exception as e:
			print(e)

