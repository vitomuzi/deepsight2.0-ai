from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from PIL import Image
import math
import time
import cv2
from scipy.misc import imread

# from utils.utils import load_pre_annt_file, save_annt_file
from models import NET

# def _predict_img(img_path, annt_file, out_file, net, out_img=None):
# 	print(img_path)
# 	class_type = ('__background__', 'abnormal')
# 	ind_to_class = dict(zip(range(len(class_type)), class_type))
# 	img = Image.open(img_path)
# 	annt_info, elemet_tree = load_annt_file(annt_file)
# 	if len(annt_info['classes_name']) == 0:
# 		annt_info_refined = {
# 			'boxes': [],
# 			'classes_name': [],
# 			'gt_ishard': []
# 		}
# 		save_annt_file(out_file, annt_info_refined)
# 		return

# 	boxes = []
# 	classes_name = []
# 	ishards = []
# 	proba = []
# 	for box in annt_info['boxes']:
# 		patch = img.crop(box.tolist())
# 		predict_result = predict(patch, net, input_size=128)
# 		predict_result = predict_result.data.cpu().numpy()
# 		predict_class = np.where(abs(predict_result-predict_result.max())<1e-10)[1][0]
# 		if predict_class == 0:
# 			continue
		
# 		boxes.append(box.tolist())
# 		classes_name.append(ind_to_class[predict_class])
# 		ishards.append(0)
# 		proba.append(predict_result.max())

# 	annt_info_refined = {
# 		'boxes': np.array(boxes).astype(np.uint16),
# 		'classes_name': classes_name,
# 		'gt_ishard': ishards
# 	}
# 	save_annt_file(out_file, annt_info_refined)
# 	if out_img != None:
# 		im_in = np.array(imread(img_path))
# 		im = im_in[:,:,::-1]
# 		im = np.copy(im)
# 		im2show = _vis_predictions(im, 'abnormal', proba, boxes, 0.5)
# 		cv2.imwrite(out_img, im2show)

def _get_img(box, img, class_name):
	xmin, ymin, xmax, ymax = box
	x_center = (xmin+xmax)/2
	y_center = (ymin+ymax)/2
	x_center = 128 if x_center<128 else x_center
	y_center = 128 if y_center<128 else y_center
	x_center = 896 if x_center>896 else x_center
	y_center = 896 if y_center>896 else y_center

	xmin = x_center - 128
	xmax = x_center + 128
	ymin = y_center - 128
	ymax = y_center + 128

	box = [xmin, ymin, xmax, ymax]

	return [box], [class_name], [np.asarray(img.crop(box))]

def get_testImg(img_np, annt_info_refined, own_data_classes):
	boxes = []
	classes = []
	patch = []
	total_probality = []
	_class_to_ind = dict(zip(own_data_classes, range(len(own_data_classes))))

	if len(annt_info_refined['classes_name']) == 0:
		# return empty list if annt file is empty
		return boxes, classes, patch, total_probality

	img = Image.fromarray(img_np)

	for idx, box_annt in enumerate(annt_info_refined['boxes']):
		box_item, box_class_item, patch_item = _get_img(box_annt, img, _class_to_ind[annt_info_refined['classes_name'][idx]])
		boxes.extend(box_item)
		classes.extend(box_class_item)
		patch.extend(patch_item)
		total_probality.append(annt_info_refined['gt_probablity'][idx])

	return boxes, classes, patch, total_probality

def interface(patches, probality_s, net, own_data_classes,cuda=True):
	ind_to_class = dict(zip(range(len(own_data_classes)), own_data_classes))

	result_flag = 0
	if len(probality_s)==0:
		result_flag = 1
		return 0, 1.0, result_flag

	mean = [0.485, 0.456, 0.406] 
	std = [0.229, 0.224, 0.225]

	if len(probality_s)<5:
	#	print('kuang  ', probality_s)
		probality_s = probality_s * (math.floor(5/len(probality_s)+1) + 1)
		patches = patches * (math.floor(5/len(patches)+1) + 1) 

	seq = np.zeros((256, 256, 3*5), dtype=np.float32)
	pro_index = np.argsort(probality_s)
	for img_index in range(1,6):
		seq_index = img_index - 1
		item_data = patches[pro_index[-img_index]]
		#img = np.array(item_data)

		#seq[:,:,seq_index*3] = item_data[:,:,0]
		#seq[:,:,seq_index*3+1] = item_data[:,:,1]
		#seq[:,:,seq_index*3+2] = item_data[:,:,2]

		seq[:,:,seq_index*3]   = (item_data[:,:,0]/255.-mean[0])/std[0]
		seq[:,:,seq_index*3+1] = (item_data[:,:,1]/255.-mean[1])/std[1]
		seq[:,:,seq_index*3+2] = (item_data[:,:,2]/255.-mean[2])/std[2]

	try:
		predict_result = predict(seq,net, cuda=cuda)
	except Exception as e:
		raise e

	predict_result = predict_result.data.cpu().numpy()
	predict_class = np.where(abs(predict_result-predict_result.max())<1e-10)[1][0]

	return predict_class, predict_result.max(), result_flag

def predict(data, net, cuda=True):
	X = torch.from_numpy(np.rollaxis(data, 2, 0)).float().unsqueeze(0)

	if cuda:
		X = Variable(X, volatile=True).cuda()
	else:
		X = Variable(X, volatile=True)

	y = F.softmax(net(X))

	if not cuda:
		y = y.cpu()
	return y
		


# def interface(img_np, annt_info, net, own_data_classes, cuda=True):
# 	ind_to_class = dict(zip(range(len(own_data_classes)), own_data_classes))

# 	img = Image.fromarray(img_np)
# 	max_index = [None for i in range(len(own_data_classes))]
# 	max_proba = [0 for i in range(len(own_data_classes))]
# 	num_boxes_array = np.zeros(len(own_data_classes))
# 	#6类 每个类15个频次 areas 0-40 40-42 42-44 44-46 46-48 48-50 50-52 52-54 54-56 56-58 58-60 60-70 70-80 80-90 90-100
# 	num_frequence_class_array = np.zeros((len(own_data_classes),15))

# 	if len(annt_info['classes_name']) == 0:
# 		annt_info_refined = {
# 			'boxes': [],
# 			'classes_name': [],
# 			'gt_ishard': [],
# 			'gt_probablity': []
# 		}
# 		return annt_info_refined, [], max_index, num_boxes_array, num_frequence_class_array

# 	boxes = []
# 	classes_name = []
# 	ishards = []
# 	proba = []
# 	for idx, box in enumerate(annt_info['boxes']):
# 		patch = img.crop(box.tolist())
# 		try:
# 			predict_result = predict(patch, net, input_size=128, cuda=cuda)
# 		except:
# 			print(box)
# 		predict_result = predict_result.data.cpu().numpy()
# 		predict_class = np.where(abs(predict_result-predict_result.max())<1e-10)[1][0]

# 		if predict_result.max() < 0.9:
# 			continue

# 		num_boxes_array[predict_class] += 1

# 		if predict_result.max() < 0.4:
# 			num_frequence_class_array[predict_class][0] += 1
# 		elif predict_result.max() < 0.42:
# 			num_frequence_class_array[predict_class][1] += 1
# 		elif predict_result.max() < 0.44:
# 			num_frequence_class_array[predict_class][2] += 1
# 		elif predict_result.max() < 0.46:
# 			num_frequence_class_array[predict_class][3] += 1
# 		elif predict_result.max() < 0.48:
# 			num_frequence_class_array[predict_class][4] += 1
# 		elif predict_result.max() < 0.50:
# 			num_frequence_class_array[predict_class][5] += 1
# 		elif predict_result.max() < 0.52:
# 			num_frequence_class_array[predict_class][6] += 1
# 		elif predict_result.max() < 0.54:
# 			num_frequence_class_array[predict_class][7] += 1
# 		elif predict_result.max() < 0.56:
# 			num_frequence_class_array[predict_class][8] += 1
# 		elif predict_result.max() < 0.58:
# 			num_frequence_class_array[predict_class][9] += 1
# 		elif predict_result.max() < 0.6:
# 			num_frequence_class_array[predict_class][10] += 1
# 		elif predict_result.max() < 0.7:
# 			num_frequence_class_array[predict_class][11] += 1
# 		elif predict_result.max() < 0.8:
# 			num_frequence_class_array[predict_class][12] += 1
# 		elif predict_result.max() < 0.9:
# 			num_frequence_class_array[predict_class][13] += 1
# 		elif predict_result.max() <= 1.0:
# 			num_frequence_class_array[predict_class][14] += 1


# 		if predict_class == 0: # or predict_class == 1
# 			if (predict_result.max() > max_proba[predict_class]) and (predict_result.max()>0.75):
# 				max_proba[predict_class] = predict_result.max()
# 				max_index[predict_class] = idx
# 			continue

# 		if (predict_result.max() > max_proba[predict_class]) and (predict_result.max()>0.75):
# 			max_proba[predict_class] = predict_result.max()
# 			max_index[predict_class] = idx
# 		boxes.append(box.tolist())
# 		classes_name.append(ind_to_class[predict_class])
# 		ishards.append(0)
# 		proba.append(predict_result.max())

# 	annt_info_refined = {
# 		# 'boxes': np.array(boxes).astype(np.uint16),
# 		'boxes': boxes,
# 		'classes_name': classes_name,
# 		'gt_ishard': ishards,
# 		'gt_probablity': proba
# 	}
# 	#print(annt_info_refined['classes_name'])
# 	return annt_info_refined, proba, max_index, num_boxes_array, num_frequence_class_array
	

def _vis_predictions(im, class_name, predictions, bndboxes, thresh=0.8):
	"""Visual debugging of detections."""
	for i in range(len(predictions)):
		bbox = bndboxes[i]
		if predictions[i] > thresh:
			cv2.rectangle(im, tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 204, 0), 2)
			cv2.putText(im, '%s: %.3f' % (class_name, predictions[i]), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
				1.0, (0, 0, 255), thickness=1)
	return im

# def predict(input_img, net, input_size=128, cuda=True):
# 	# rescale image.
# 	scale_factor = min(input_size/float(input_img.size[0]), input_size/float(input_img.size[1]))
# 	img = input_img.resize([min(input_size, math.ceil(input_img.size[0]*scale_factor)), min(input_size, math.ceil(input_img.size[1]*scale_factor))], resample=Image.BILINEAR)
# 	data = np.asarray(img)
# 	data_padding = np.zeros((input_size, input_size, 3))
# 	w_head = math.floor((input_size-img.size[0])/2.)
# 	h_head = math.floor((input_size-img.size[1])/2.)
# 	data_padding[h_head:h_head+img.size[1], w_head:w_head+img.size[0], :] = data

# 	# process the rescaled image.
# 	X = torch.FloatTensor(np.rollaxis(data_padding, 2, 0)).unsqueeze(0)

# 	if cuda:
# 		X = Variable(X, volatile=True).cuda()
# 	else:
# 		X = Variable(X, volatile=True)

# 	y = F.softmax(net(X))

# 	if not cuda:
# 		y = y.cpu()
# 	return y


class WrappedModel(nn.Module):
	def __init__(self, net):
		super(WrappedModel, self).__init__()
		self.module = net # that I actually define.
	def forward(self, x):
		return self.module(x)


def load_model(net_path, own_data_classes, cuda=True):
	net = NET.DenseNet(
				num_classes=len(own_data_classes),
				depth=46,
				growthRate=12,
				compressionRate=2,
				dropRate=0
				)
	net = torch.nn.DataParallel(net)
	
	print("Model {} loading...".format(net_path))
	if cuda:
		checkpoint = torch.load(net_path)
		net.load_state_dict(checkpoint)
	else:
		checkpoint = torch.load(net_path, map_location=(lambda storage, loc: storage))
		net.load_state_dict(checkpoint)


	if cuda:
		net.cuda()

	return net


if __name__ == '__main__':
	net_path = '/mnt/data/zhouming/cc/densenet_step3/models/CP191.pth'
	class__s = ['a','b'] 
	net = load_model(net_path,class__s)

	# im_in = np.array(imread('data/own_data/test/JISCZWFY1802271000040030-3.jpg'))
	# annt_info, elemet_tree = load_annt_file('data/own_data/JISCZWFY1802271000040030-3.xml')

	# annt_info_refined = interface(im_in, annt_info, net, cuda=True)

	# save_annt_file('data/own_data/JISCZWFY1802271000040030-3-refine.xml', annt_info_refined)
