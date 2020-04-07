from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
#import torchvision.models as modules
import torch.nn.functional as F

import random
import argparse
import math
import numpy as np

from utilities.patch_db import patch_db
from utilities.normal_cell_db import normal_cell_db
from utilities.db_interface import db_interface, sampler_interface
from models import densenet
from models import densenet161
from models.focal_loss import FocalLoss

import os
from tensorboardX import SummaryWriter

def _calc_precision(y, y_pred, label):
	gt = y[y_pred==label]
	return torch.sum(gt==label), len(gt)

def _calc_recall(y, y_pred, label):
	pred = y_pred[y==label]
	return torch.sum(pred==label), len(pred)

# get alpha list for focal loss
def _get_alpha(ratio_list):
	reciprocal_ratio_list = [1.0/item for item in ratio_list]
	sum_reciprocal_ratio_list = sum(reciprocal_ratio_list)
	alpha_list = [item/sum_reciprocal_ratio_list for item in reciprocal_ratio_list]
	return alpha_list

def _split_dataset(db, train_prop=0.8):
	num_data = len(db)
	split_index = math.floor(len(db)*train_prop)
	random.shuffle(db)

	return db[:split_index], db[split_index:]

def _append_db(train_total, val_total, ratio_list_train, ratio_list_val, db):
	train, val = _split_dataset(db)

	train_total += train
	val_total += val
	ratio_list_train.append(len(train))
	ratio_list_val.append(len(val))

# generate data loader
def _get_dataloader(batch_size, random_seed=None):
	# load data by compare anchor xml files with lable xml files
	cfg_path = 'cfgs/data_cfg_neg.yml'
	db_neg = normal_cell_db(cfg_path, cache_dir='data/cache').normaldb # 0
	db_neg_sampled = random.sample(db_neg, 60000)
	print('{} bndboxes in db_neg'.format(len(db_neg)))

	cfg_path = 'cfgs/data_cfg_cell.yml'
	db_cell = normal_cell_db(cfg_path, cache_dir='data/cache').normaldb
	db_normal = list(filter(lambda item:item['class']==0, db_cell))
	db_abnormal = list(filter(lambda item:item['class']==1, db_cell))
	db_normal_sampled = random.sample(db_normal, 100000)
	print('{} bndboxes in db_normal'.format(len(db_normal)))

	db_normal = db_normal_sampled + db_neg_sampled
	print('{} bndboxes in db_normal used'.format(len(db_normal)))
	print('{} bndboxes in db_abnormal'.format(len(db_abnormal)))
	
	# load data from mono lable xml files
	cfg_path = 'cfgs/data_cfg_H-SIL.yml'
	db_H_SIL = patch_db(cfg_path, cache_dir='data/cache', aug_factor=5).patchdb
	print('{} bndboxes in db_H_SIL'.format(len(db_H_SIL)))

	cfg_path = 'cfgs/data_cfg_Candida.yml'
	db_Candida = patch_db(cfg_path, cache_dir='data/cache', aug_factor=5).patchdb
	print('{} bndboxes in db_Candida'.format(len(db_Candida)))

	cfg_path = 'cfgs/data_cfg_fungi.yml'
	db_fungi = patch_db(cfg_path, cache_dir='data/cache', aug_factor=50).patchdb
	print('{} bndboxes in db_fungi'.format(len(db_fungi)))

	cfg_path = 'cfgs/data_cfg_Herpesviruses.yml'
	db_Herpesviruses = patch_db(cfg_path, cache_dir='data/cache', aug_factor=300).patchdb
	print('{} bndboxes in db_Herpesviruses'.format(len(db_Herpesviruses)))

	db_total = db_normal  + db_abnormal + db_H_SIL + db_Candida + db_fungi + db_Herpesviruses 

	random.shuffle(db_total)
	print('{} bndboxes in db_total'.format(len(db_total)))

	dataset = db_interface(db_total)

	# split as train and val
	ratio_list_train, ratio_list_val, train_total, val_total = [], [], [], []
	_append_db(train_total, val_total, ratio_list_train, ratio_list_val, db_normal)
	# train, val = _split_dataset(db_neg, train_prop=0.8)
	# train_total += train
	# val_total += val
	# ratio_list_train[0] = ratio_list_train[0] + len(train)
	# ratio_list_val[0] = ratio_list_val[0] + len(val)  # 0

	_append_db(train_total, val_total, ratio_list_train, ratio_list_val, db_abnormal) # 1
	_append_db(train_total, val_total, ratio_list_train, ratio_list_val, db_H_SIL) # 2
	# _append_db(train_total, val_total, ratio_list_train, ratio_list_val, db_Candida) # 3
	# _append_db(train_total, val_total, ratio_list_train, ratio_list_val, db_fungi) # 4
	# _append_db(train_total, val_total, ratio_list_train, ratio_list_val, db_Herpesviruses) # 5

	if random_seed!=None:
		sampler_train = sampler_interface(len(train_total), random_seed=random_seed*10)
		sampler_val = sampler_interface(len(val_total), random_seed=random_seed*10)
	else:
		sampler_train = sampler_interface(len(train_total))
		sampler_val = sampler_interface(len(val_total))
		
	train_loader = torch.utils.data.DataLoader(db_interface(train_total), batch_size=batch_size, sampler=sampler_train, num_workers=2)
	val_loader = torch.utils.data.DataLoader(db_interface(val_total), batch_size=batch_size, sampler=sampler_val, num_workers=2)

	return train_loader, ratio_list_train, val_loader, ratio_list_val



def train(args, net, epoch, train_loader, gpu=True):
	net.train()
	
	data_itr = iter(train_loader)

	epoch_loss = 0
	epoch_err = 0
	epoch_tp, epoch_fp, epoch_fn = 0, 0, 0

	for index_batch, (img, label) in enumerate(data_itr):
		if gpu:
			X = Variable(img).cuda()
			y = Variable(label).cuda()
		else:
			X = Variable(img)
			y = Variable(label)

		predict = net(X)
		loss = args.criterion(predict, y)
		epoch_loss += loss.data[0]

		args.optimizer.zero_grad()
		loss.backward()
		args.optimizer.step()

		y_predict = F.softmax(predict)
		y_predict = torch.max(y_predict.data, dim=1)[1]
		epoch_err += torch.sum(y_predict!=y.data)
		batch_err = 1.*torch.sum(y_predict!=y.data)/len(y_predict)
		tp, fp = _calc_precision(y.data, y_predict, 1)
		tp, fn = _calc_recall(y.data, y_predict, 1)
		epoch_tp += tp
		epoch_fp += fp
		epoch_fn += fn

		if index_batch%(math.ceil(len(train_loader)/10)) == 0:
			print('{0:.4f} --- loss: {1:.6f}\t err: {2:.6f}'.format((index_batch+1)/len(train_loader), loss.data[0], batch_err))

	args.writer.add_scalar('train_loss', epoch_loss, epoch)
	args.writer.add_scalar('train_err', epoch_err/args.num_data['train'], epoch)
	args.writer.add_scalar('train_precision', 1.*epoch_tp/(epoch_tp+epoch_fp), epoch)
	args.writer.add_scalar('train_recall', 1.*epoch_tp/(epoch_tp+epoch_fn), epoch)

	print('Epoch {0:03d} train - Loss: {1:.6f},\t Err: {2:.6f}'.format(
		epoch+1, epoch_loss/index_batch, epoch_err/args.num_data['train']))
	if epoch%10 == 0:
		torch.save(net.state_dict(), args.save + '/CheckPoint{}.pth'.format(epoch+1))
			

def val(args, net, epoch, val_loader, gpu=True):
	net.eval()
	
	data_itr = iter(val_loader)

	epoch_loss = 0
	epoch_err = 0
	epoch_tp, epoch_fp, epoch_fn = 0, 0, 0

	for index_batch, (img, label) in enumerate(data_itr):
		if gpu:
			X = Variable(img, volatile=True).cuda()
			y = Variable(label, volatile=True).cuda()
		else:
			X = Variable(img, volatile=True)
			y = Variable(label, volatile=True)

		predict = net(X)
		loss = args.criterion(predict, y)
		epoch_loss += loss.data[0]

		y_predict = F.softmax(predict)
		y_predict = torch.max(y_predict.data, dim=1)[1]
		epoch_err += torch.sum(y_predict!=y.data)
		tp, fp = _calc_precision(y.data, y_predict, 1)
		tp, fn = _calc_recall(y.data, y_predict, 1)
		epoch_tp += tp
		epoch_fp += fp
		epoch_fn += fn

	args.writer.add_scalar('val_loss', epoch_loss, epoch)
	args.writer.add_scalar('val_err', epoch_err/args.num_data['val'], epoch)
	args.writer.add_scalar('val_precision', 1.*epoch_tp/(epoch_tp+epoch_fp), epoch)
	args.writer.add_scalar('val_recall', 1.*epoch_tp/(epoch_tp+epoch_fn), epoch)

	print('Epoch {0:03d} val - Loss: {1:.6f},\t Err: {2:.6f}'.format(
		epoch+1, epoch_loss/index_batch, epoch_err/args.num_data['val']))
	return epoch_loss/index_batch


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train a DenseNet.')
	parser.add_argument('--batch_size', dest='batch_size', help='batch size', type=int, default=5)
	parser.add_argument('--lr', dest='lr', help='batch size', type=float, default=0.01)
	parser.add_argument('--epoch', dest='epoch', help='num epoch', type=int, default=150)
	parser.add_argument('--save', dest='save', help='check_point_dir', type=str, default='data/models')
	parser.add_argument('--loss', dest='loss', help='loss function', type=str, default='focal')

	args = parser.parse_args()

	# init densenet161
	gpu = True
	net161 = densenet161.densenet161(pretrained=False, num_classes=3, drop_rate=0.2)
	# net161 = modules.DenseNet(
	# 	growth_rate=48, 
	# 	block_config=(6, 12, 36, 24), 
	# 	num_init_features=96, 
	# 	bn_size=4, 
	# 	drop_rate=0, 
	# 	num_classes=3
	# 	)
	# #net161.load_state_dict('data/models_pretrained/densenet161-8d451a50.pth')

	# # init net
	# gpu = True
	# net = densenet.DenseNet(
	# 		num_classes=6,
	# 		depth=46,
	# 		growthRate=12,
	# 		compressionRate=2,
	# 		dropRate=0
	# 		)

	if gpu:
		net161 = torch.nn.DataParallel(net161).cuda()

	print('''
	Starting training:
		Epochs: {}
		Batch size: {}
		Checkpoints: {}
		CUDA: {}
	'''.format(args.epoch, args.batch_size, args.save, str(gpu)))

	if not os.path.exists(args.save):
		os.makedirs(args.save)
	
	# init optimizer
	args.optimizer = optim.SGD(net161.parameters(), lr=args.lr, momentum=0.85, weight_decay=1e-8)
	args.scheduler = lr_scheduler.ReduceLROnPlateau(args.optimizer, 'min', patience=10, factor=0.1)

	# load data
	train_loader, train_ratio_list, val_loader, val_ratio_list = _get_dataloader(batch_size=args.batch_size, random_seed=1)
	alpha_list = _get_alpha(train_ratio_list)
	#alpha_list = [1, 5, 20, 1, 1, 1]
	alpha_list = [1, 8, 32]
	args.num_data = {'train':float(sum(train_ratio_list)), 'val':float(sum(val_ratio_list))}
	if args.loss == 'focal':
		args.criterion = FocalLoss(gamma=3, alpha=alpha_list)
	elif args.loss == 'nll':
		weight = torch.FloatTensor(alpha_list).cuda() if gpu else torch.FloatTensor(alpha_list)
		args.criterion = lambda x,y,weight=weight: nn.NLLLoss(weight=weight)(F.log_softmax(x),y)
	else:
		print('undefined loss')
	#args.criterion = FocalLoss(gamma=3, alpha=alpha_list)

	args.writer = SummaryWriter(args.save)
	for epoch in range(1, args.epoch):
		train(args, net161, epoch, train_loader, gpu=True)
		val_loss = val(args, net161, epoch, val_loader, gpu=True)
		args.scheduler.step(val_loss)


	args.writer.close()