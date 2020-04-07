from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torchvision import transforms
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import argparse
import math
import numpy as np

import densenet
import os


def load_model(net_path, own_data_classes, cuda=True):
	net = densenet.DenseNet(
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

def predict(input_img, net, cuda=True):
	# rescale image.
#	scale_factor = min(input_size/float(input_img.size[0]), input_size/float(input_img.size[1]))
#	img = input_img.resize([min(input_size, math.ceil(input_img.size[0]*scale_factor)), min(input_size, math.ceil(input_img.size[1]*scale_factor))], resample=Image.BILINEAR)
#	data = np.asarray(img)
#	data_padding = np.zeros((input_size, input_size, 3))
#	w_head = math.floor((input_size-img.size[0])/2.)
#	h_head = math.floor((input_size-img.size[1])/2.)
#	data_padding[h_head:h_head+img.size[1], w_head:w_head+img.size[0], :] = data
	normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])

	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
		normalize
    ])

	img = transform(input_img)
	# process the rescaled image.
	#X = torch.FloatTensor(np.rollaxis(data_padding, 2, 0)).unsqueeze(0)
	X = img.unsqueeze(0)

	if cuda:
		X = Variable(X, volatile=True).cuda()
	else:
		X = Variable(X, volatile=True)

	y = F.softmax(net(X))

	if not cuda:
		y = y.cpu()
	return y

def interface(img_np, net, cuda=True):
	img = Image.fromarray(img_np)
	#try:	
		#global predict_resulti
		#predict_resulti = predict(img,net,cuda=cuda)
	predict_result = predict(img,net,cuda=cuda)
	#except:
	#	print('error')
	predict_ = torch.max(predict_result.data, dim=1)[1]
	return predict_[0]


