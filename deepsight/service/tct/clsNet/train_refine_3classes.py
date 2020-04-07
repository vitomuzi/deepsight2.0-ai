from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import argparse
import math

from utilities.refine_db import refine_db
from utilities.normal_cell_db import normal_cell_db
from utilities.db_interface import db_interface, sampler_interface
from models import densenet

import os
from tensorboardX import SummaryWriter

def _calc_precision(y, y_pred, label):
	gt = y[y_pred==label]
	return torch.sum(gt==label), len(gt)

def _calc_recall(y, y_pred, label):
	pred = y_pred[y==label]
	return torch.sum(pred==label), len(pred)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         print('init net')
#         nn.init.kaiming_normal(m.weight)
#         m.bias.data.zero_()

# sample data to balance data size of different classses
def _sample_db(input_db, random_seed=None, classes=(0,1)):
	background_db = list(filter(lambda item:item['class']==0, input_db))
	abnormal_db = list(filter(lambda item:item['class']==1, input_db))
	if random_seed!=None:
		random.seed(random_seed)
	background_db_sampled = random.sample(background_db, len(abnormal_db))
	balance_db = background_db_sampled + abnormal_db
	random.shuffle(balance_db)
	return balance_db

# generate data loader
# for the inbalence of data size, negtive data is sampled to be equal to postive data 
def _get_dataloader(batch_size, random_seed=None, refine_target='all_neg'):
	# data from negtive data
	cfg_path = 'cfgs/data_cfg_neg.yml'
	db_raw = normal_cell_db(cfg_path, cache_dir='data/cache')
	neg_db = db_raw.normaldb
	print('{} bndboxes in neg_db'.format(len(neg_db)))
	#neg_db = random.sample(neg_db, 30000)
	print('{} bndboxes in neg_db used'.format(len(neg_db)))

	# data from postive data
	cfg_path = 'cfgs/data_cfg_refine_db_3classes.yml'
	db_raw = refine_db(cfg_path, 'refine_db_3classes', cache_dir='data/cache')
	db = db_raw.refinedb
	background_db = list(filter(lambda item:item['class']==0, db))
	abnormal_db = list(filter(lambda item:item['class']==1, db))
	HSIL_db = list(filter(lambda item:item['class']==2, db))
	print('{} bndboxes in background_db'.format(len(background_db)))
	print('{} bndboxes in abnormal_db'.format(len(abnormal_db)))
	print('{} bndboxes in HSIL_db'.format(len(HSIL_db)))

	if random_seed!=None:
		random.seed(random_seed)

	if refine_target=='all_neg':
		if len(neg_db) < len(abnormal_db):
			# sample abnormal_db to balance data 
			# abnormal_db_sampled = random.sample(abnormal_db, len(neg_db))
			# balance_db = abnormal_db_sampled + neg_db
			# duplicate neg_db
			neg_db_ext = neg_db * math.floor(len(abnormal_db)/len(neg_db))
			balance_db = abnormal_db + neg_db_ext
		else:
			print('no implementation yet')

	elif refine_target=='all':
		if len(abnormal_db) > len(neg_db): 
			background_db_sampled = random.sample(background_db, len(abnormal_db)-len(neg_db))
			balance_db = background_db_sampled + abnormal_db + neg_db
		else:
			#abnormal_db_ext = abnormal_db * math.floor((len(background_db) + len(neg_db))/len(abnormal_db))
			abnormal_db_ext = abnormal_db * 6
			HSIL_db_ext = HSIL_db * 30 
			balance_db = abnormal_db_ext + HSIL_db_ext + background_db + neg_db 
			print('{} bndboxes in abnormal_db used'.format(len(abnormal_db_ext)))
	else:
		print('wrong refine_target value')
		return

	random.shuffle(balance_db)
	print('{} bndboxes in balance_db'.format(len(balance_db)))
	dataset = db_interface(balance_db)
	if random_seed!=None:
		sampler = sampler_interface(len(balance_db), random_seed=random_seed*10)
	else:
		sampler = sampler_interface(len(balance_db))
		
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

	return dataloader, len(balance_db)

# train net
def adjust_learning_rate(optimizer, decay=0.1):
	for param_group in optimizer.param_groups:
		param_group['lr'] = decay * param_group['lr']

def train(writer, net, epochs=300, batch_size=5, lr_decay_period=100, db_reload_period=50, refine_target='all_neg', check_point_dir='data/models', gpu=True):
	lr = 0.01
	train_total = 0
	if gpu:
		net = torch.nn.DataParallel(net).cuda()

	net.train()
	# net.apply(weights_init)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.85, weight_decay=0.0005)

	dataloader, train_total = _get_dataloader(batch_size=batch_size, random_seed=1, refine_target=refine_target)

	print('''
	Starting training:
		Epochs: {}
		Batch size: {}
		Checkpoints: {}
		CUDA: {}
	'''.format(epochs, batch_size, check_point_dir, str(gpu)))

	for epoch in range(1, epochs+1):
		epoch_loss = 0
		if epoch % db_reload_period == 0:
			dataloader, train_total = _get_dataloader(batch_size=batch_size, random_seed=epoch, refine_target=refine_target)
		data_itr = iter(dataloader)
		if epoch%lr_decay_period == 0:
			adjust_learning_rate(optimizer)

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
			loss = criterion(predict, y)
			epoch_loss += loss.data[0]

			# if index_batch%(math.ceil(len(dataloader)/10)) == 0:
			# 	print('{0:.4f} --- loss: {1:.6f}'.format((index_batch+1)/len(dataloader), loss.data[0]))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			y_predict = F.softmax(predict)
			y_predict = torch.max(y_predict.data, dim=1)[1]
			epoch_err += torch.sum(y_predict!=y.data)
			batch_err = 1.*torch.sum(y_predict!=y.data)/len(y_predict)
			tp, fp = _calc_precision(y.data, y_predict, 1)
			tp, fn = _calc_recall(y.data, y_predict, 1)
			epoch_tp += tp
			epoch_fp += fp
			epoch_fn += fn

			if index_batch%(math.ceil(len(dataloader)/10)) == 0:
				print('{0:.4f} --- loss: {1:.6f}\t err: {2:.6f}'.format((index_batch+1)/len(dataloader), loss.data[0], batch_err))

		writer.add_scalar('train_loss', epoch_loss, epoch)
		writer.add_scalar('train_err', epoch_err/train_total, epoch)
		writer.add_scalar('train_precision', 1.*epoch_tp/(epoch_tp+epoch_fp), epoch)
		writer.add_scalar('train_recall', 1.*epoch_tp/(epoch_tp+epoch_fn), epoch)

		print('Epoch {} finished ! Loss: {}'.format(epoch+1, epoch_loss/index_batch))
		if epoch%10 == 0:
			torch.save(net.state_dict(), check_point_dir + '/CP{}.pth'.format(epoch+1))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train a DenseNet.')
	parser.add_argument('--batch_size', dest='batch_size', help='batch size', type=int, default=5)
	parser.add_argument('--epoch', dest='epoch', help='num epoch', type=int, default=300)
	parser.add_argument('--lr_decay_period', dest='lr_decay_period', help='lr decaies after lr_decay_period epochs', type=int, default=100)
	parser.add_argument('--db_reload_period', dest='db_reload_period', help='reload data after db_reload_period epochs', type=int, default=50)
	parser.add_argument('--refine_all', dest='refine_all', help='refine all_neg or all participants', action='store_true', default=False)
	parser.add_argument('--save', dest='save', help='check_point_dir', type=str, default='data/models')

	args = parser.parse_args()
	batch_size = args.batch_size
	epoch = args.epoch
	lr_decay_period = args.lr_decay_period
	db_reload_period = args.db_reload_period
	refine_all = args.refine_all
	check_point_dir = args.save

	if refine_all:
		refine_target='all'
	else:
		refine_target='all_neg'

	gpu = True
	net = densenet.DenseNet(
			num_classes=3,
			depth=46,
			growthRate=12,
			compressionRate=2,
			dropRate=0
			)
	writer = SummaryWriter(args.save)
	train(writer, net, epochs=epoch, batch_size=batch_size, lr_decay_period=lr_decay_period, db_reload_period=db_reload_period, refine_target=refine_target, check_point_dir=check_point_dir, gpu=True)

	writer.close()