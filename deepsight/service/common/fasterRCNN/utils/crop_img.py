import numpy as np
import math
import os
from PIL import Image

import general as ut

# crop large original image into some smaller images.
def crop_img_2d(img_path, out_dir, sub_array=[3, 2]):
	get_sub_path = lambda sub_index, ori_path, out_dir:out_dir + '/' + os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1]

	img_ori = Image.open(img_path)
	img_size = img_ori.size
	grid_array = ([0]+[math.floor((i+1)*img_size[0]/float(sub_array[0])) for i in range(sub_array[0])], [0]+[math.floor((i+1)*img_size[1]/float(sub_array[1])) for i in range(sub_array[1])])
	
	sub_index = 0
	for sub_index_x in range(sub_array[0]):
		for sub_index_y in range(sub_array[1]):
			sub_img_path = get_sub_path(sub_index, img_path, out_dir)
			x_min_img = grid_array[0][sub_index_x]
			y_min_img = grid_array[1][sub_index_y]
			x_max_img = grid_array[0][sub_index_x+1]-1
			y_max_img = grid_array[1][sub_index_y+1]-1
			sub_img = img_ori.crop((x_min_img, y_min_img, x_max_img+1, y_max_img+1))
			sub_img.save(sub_img_path)
			sub_index += 1

# crop large original image into some smaller images with a n*m grid. corresponding annotation files are also generated.
def _crop_img_2d(img_path, annt_path, out_dir, sub_array=[3, 2]):
	get_sub_path = lambda sub_index, ori_path, out_dir:out_dir + '/' + os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1]

	img_ori = Image.open(img_path)
	img_size = img_ori.size
	grid_array = ([0]+[math.floor((i+1)*img_size[0]/float(sub_array[0])) for i in range(sub_array[0])], [0]+[math.floor((i+1)*img_size[1]/float(sub_array[1])) for i in range(sub_array[1])])
	
	annt_info, template_tree = ut.load_annt_file(annt_path)
	sub_index = 0
	for sub_index_x in range(sub_array[0]):
		for sub_index_y in range(sub_array[1]):
			sub_img_path = get_sub_path(sub_index, img_path, out_dir)
			sub_annt_path = get_sub_path(sub_index, annt_path, out_dir)
			x_min_img = grid_array[0][sub_index_x]
			y_min_img = grid_array[1][sub_index_y]
			x_max_img = grid_array[0][sub_index_x+1]-1
			y_max_img = grid_array[1][sub_index_y+1]-1
			img_box = np.array([x_min_img, y_min_img, x_max_img, y_max_img])
			objs_info = _prune_bndbox(annt_info, img_box)
			ut.save_annt_file(sub_annt_path, objs_info, template_tree)
			sub_img = img_ori.crop((x_min_img, y_min_img, x_max_img+1, y_max_img+1))
			sub_img.save(sub_img_path)
			sub_index += 1

# transfer box coordinates to the new image space, and prune unquanlified bounding boxes for cropped images.
def _prune_bndbox(objs_info, img_box):
	# transfer coordinate space
	x_ori = img_box[0]
	y_ori = img_box[1]
	img_width = img_box[2] - img_box[0] + 1
	img_height = img_box[3] - img_box[1] + 1
	objs_boxes = objs_info['boxes'].astype(np.float32)
	new_boxes = np.zeros(objs_info['boxes'].shape, dtype=np.float32)
	new_boxes[:, 0] = objs_boxes[:, 0] - x_ori
	new_boxes[:, 2] = objs_boxes[:, 2] - x_ori
	new_boxes[:, 1] = objs_boxes[:, 1] - y_ori
	new_boxes[:, 3] = objs_boxes[:, 3] - y_ori

	# prune boxes
	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	gt_classes = []
	ishards = np.empty(shape=[0], dtype=np.int32)
	for i in range(len(objs_info['boxes'])):
		x1 = new_boxes[i, 0]
		y1 = new_boxes[i, 1]
		x2 = new_boxes[i, 2]
		y2 = new_boxes[i, 3]
		remove_flag, new_box = _bndbox_remove_flag(box=[x1, y1, x2, y2], shape_img=[img_width, img_height], ratio_lim=0.2, IOU_lim=0.55)
		if remove_flag:
		    continue

		difficult = objs_info['gt_ishard'][i]
		ishards = np.append(ishards, difficult)
		boxes = np.append(boxes, np.expand_dims(new_box, axis=0), axis=0)
		gt_classes.append(objs_info['gt_classes_name'][i])

	return {'boxes': boxes,
			'gt_classes_name': gt_classes,
			'gt_ishard': ishards}

def _bndbox_remove_flag(box, shape_img, ratio_lim, IOU_lim):
	new_box = np.zeros(4, dtype=np.uint16)
	new_box[0] = _new_cord(box[0], shape_img[0])
	new_box[1] = _new_cord(box[1], shape_img[1])
	new_box[2] = _new_cord(box[2], shape_img[0])
	new_box[3] = _new_cord(box[3], shape_img[1])
	IOU = (new_box[3]-new_box[1]+1)*(new_box[2]-new_box[0]+1)/((box[3]-box[1]+1)*(box[2]-box[0]+1))

	return new_box[0]>=new_box[2] or new_box[1]>=new_box[3] or not (filter_bndbox(new_box, ratio=ratio_lim)) or IOU<IOU_lim, new_box

def filter_bndbox(bndbox, ratio=0.2):
	x_min, y_min, x_max, y_max = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
	if x_max-x_min<5:
		return False
	elif y_max-y_min<5:
		return False
	elif float(x_max-x_min)/(y_max-y_min) < ratio:
		return False
	elif float(x_max-x_min)/(y_max-y_min) > 1/ratio:
		return False
	else:
		return True

def _new_cord(cord, length):
	func_in_flag = lambda cord, length: 0<=cord and cord<length
	if func_in_flag(cord, length):
		new_cord = cord
	elif cord<0:
		new_cord = 0
	else:
		new_cord = length-1
	return new_cord


# crop images with the sliding window to eliminate the effect of crop on the border areas
def _crop_img_slide_2d(img_path, annt_path, out_dir, sub_array=[3, 2]):
	get_sub_path = lambda sub_index, ori_path, out_dir:out_dir + '/' + os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1]

	img_ori = Image.open(img_path)
	img_size = img_ori.size
	sub_size = [math.floor(img_size[idx]/float(num_itm)) for idx, num_itm in enumrate(sub_array)]
	sub_array = [num_itm*2-1 for num_itm in sub_array]
	grid_array = ([math.floor(i*img_size[0]/float(sub_array[0]+1)) for i in range(sub_array[0])],
					[math.floor(i*img_size[1]/float(sub_array[1]+1)) for i in range(sub_array[1])])
	
	annt_info, template_tree = ut.load_annt_file(annt_path)
	sub_index = 0
	for sub_index_x in range(sub_array[0]):
		for sub_index_y in range(sub_array[1]):
			sub_img_path = get_sub_path(sub_index, img_path, out_dir)
			sub_annt_path = get_sub_path(sub_index, annt_path, out_dir)
			x_min_img = grid_array[0][sub_index_x]
			y_min_img = grid_array[1][sub_index_y]
			x_max_img = grid_array[0][sub_index_x] + sub_size[0]
			y_max_img = grid_array[1][sub_index_y] + sub_size[1]
			img_box = np.array([x_min_img, y_min_img, x_max_img, y_max_img])
			objs_info = _prune_bndbox(annt_info, img_box)
			ut.save_annt_file(sub_annt_path, objs_info, template_tree)
			sub_img = img_ori.crop((x_min_img, y_min_img, x_max_img+1, y_max_img+1))
			sub_img.save(sub_img_path)
			sub_index += 1



if __name__ == '__main__':
	# for faster rcnn================
	_dir = '/home/zhouming/data/CC/process7500_croped/3072*2048/TANG20180703'
	annt_dir = _dir
	img_dir = _dir
	out_dir = '/home/zhouming/data/CC/process7500_croped/3072_afterCrop'
	file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')
	
	for file_name in file_itr:
		annt_path = annt_dir+'/'+file_name+'.xml'
		img_path = img_dir+'/'+file_name+'.jpg'
		print(img_path)
		_crop_img_2d(img_path, annt_path, out_dir)
		#crop_img_2d(img_path, out_dir)

	# # for mask rcnn==================
	# _dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/train_raw'
	# annt_dir = _dir
	# img_dir = _dir
	# out_dir = '/home/lc/code/Mask_RCNN/data/own_data'
	# file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')
	
	# for file_name in file_itr:
	# 	annt_path = annt_dir+'/'+file_name+'.xml'
	# 	img_path = img_dir+'/'+file_name+'.jpg'
	# 	print(img_path)
	# 	_crop_img_2d(img_path, annt_path, out_dir, sub_array=[6, 4])
	# 	# crop_img_2d(img_path, out_dir)
