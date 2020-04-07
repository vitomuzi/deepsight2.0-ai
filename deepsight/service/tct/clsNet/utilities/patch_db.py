from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
from PIL import Image, ImageDraw
import yaml
import random
import math

# whether work as a module
try:
    from utils import get_file_list, load_annt_file
except ImportError:
    from .utils import get_file_list, load_annt_file

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class patch_db(object):
	def __init__(self, cfg_path, cache_dir='data/cache', aug_factor=None, class_name=None):

		with open(cfg_path, 'r') as f:
			data_cfg = yaml.load(f)
		f.close()
		print(cfg_path)

		self._img_dir = data_cfg['data_path']
		self._gt_dir = data_cfg['data_path']
		self._db_name = data_cfg['db_name']
		self._suffix = data_cfg['image_ext']
		self._classes = data_cfg['classes']
		self._class_name = class_name

		self._cache_dir = cache_dir
		self._num_classes = len(self._classes)
		self._class_to_ind = data_cfg['class_to_ind']
		self._aug_factor = aug_factor

		self._patchdb = self._patch_database()

	@property
	def patchdb(self):
		return self._patchdb

	# generate patch database
	def _patch_database(self):
		# load cached data if it existed.
		cache_file = os.path.join(self._cache_dir, self._db_name+'_gt_db.pkl')
		if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				patchdb = pickle.load(fid)
			print('{} database loaded from {}'.format(self._db_name, cache_file))
			return patchdb

        # file list is generated from predict xml files
		file_name_itr = get_file_list(self._gt_dir, file_suffix=self._suffix)
		db_boxes = []
		db_class = []
		patch_img = []
		img_path_list = []
		for file_name in file_name_itr:
			# filter the required files
			if not (self._db_name in file_name):
				continue
			print('processing ' + file_name)
			annt_file = os.path.join(self._gt_dir, file_name+'.xml')
			img_file = os.path.join(self._img_dir, file_name+self._suffix)
			try:
				img = Image.open(img_file)
				boxes, classes, patches = self._get_box_info(annt_file, img)
			except ValueError as e:
				print(e)
				classes = []
			except OSError as e:
				print(e)
				classes = []

			# continue if the annt file is empty
			if len(classes)==0:
				continue
			
			# add box to db
			img_path_list.extend([os.path.join(self._img_dir, file_name+self._suffix)]*len(classes)) 
			db_boxes.extend(boxes)
			db_class.extend(classes)
			patch_img.extend(patches)

		db_info = zip(db_boxes, db_class, img_path_list, patch_img)
		db = [dict(zip(('box', 'class', 'path', 'image'), item_info)) for item_info in db_info]
		with open(cache_file, 'wb') as fid:
			pickle.dump(db, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote {} database to {}'.format(self._db_name, cache_file))
		return db

    # get annotation info
	def _get_box_info(self, annt_file, img):
		if not(os.path.exists(annt_file)):
			print(''''{} doesn\'t exist.'''.format(annt_file))
			return [], [], []
		else:
			annt_info, elemet_tree = load_annt_file(annt_file)
		boxes = []
		classes = []
		patch = []
		if len(annt_info['classes_name']) == 0:
			# return empty list if annt file is empty
			return boxes, classes, patch

		for idx, box_annt in enumerate(annt_info['boxes']):
			if not (self._class_name is None) and annt_info['classes_name'][idx] != self._class_name:
				continue
			box_item, box_class_item, patch_item = self._get_bndboxes(box_annt, img, self._class_to_ind[annt_info['classes_name'][idx]])
			boxes.extend(box_item)
			classes.extend(box_class_item)
			patch.extend(patch_item)

		# return boxes, classes, patch
		return boxes, classes, patch

	def _get_bndboxes(self, box, img, class_name):
		if self._aug_factor is None:
			return [box.tolist()], [class_name], [np.asarray(img.crop(box.tolist()))]

		gt_box = box
		boxes = []
		patches = []
		for _ in range(self._aug_factor):
			box, patch = self._aug_bndboxes(gt_box, img)
			boxes.append(box)
			patches.append(patch)

		class_names = [class_name] * len(boxes)

		return boxes, class_names, patches


	def _aug_bndboxes(self, gt_box, img):
		xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_box
		gt_width = xmax_gt - xmin_gt
		gt_height = ymax_gt - ymin_gt
		gt_center = ((xmax_gt+xmin_gt)//2, (ymax_gt+ymin_gt)//2)
		shift_percent = 0.2
		scale_percent = 0.2
		# random crop is implemented
		while True:
			bndbox_width = math.floor(gt_width * random.uniform(1-scale_percent,1+scale_percent))
			bndbox_height = math.floor(gt_height * random.uniform(1-scale_percent,1+scale_percent))
			bndbox_center = (
				math.floor(gt_center[0] + gt_width * random.uniform(-shift_percent,shift_percent)),
				math.floor(gt_center[1] + gt_height * random.uniform(-shift_percent,shift_percent))
				)
			xmin_box = 0 if bndbox_center[0] - bndbox_width//2 < 0 else bndbox_center[0] - bndbox_width//2
			ymin_box = 0 if bndbox_center[1] - bndbox_height//2 < 0 else bndbox_center[1] - bndbox_height//2
			xmax_box = img.size[0] if xmin_box + bndbox_width > img.size[0] else xmin_box + bndbox_width
			ymax_box = img.size[1] if ymin_box + bndbox_height > img.size[1] else ymin_box + bndbox_height
			box = [xmin_box, ymin_box, xmax_box, ymax_box]
			if self._get_overlap(box, gt_box) > 0.5:
				break

		patch = np.asarray(img.crop(box))

		return box, patch

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

	cfg_path = '../cfgs/data_cfg_H-SIL.yml'

	db = patch_db(cfg_path, cache_dir='data/cache', aug_factor=50)
	test = patch_db.patchdb

