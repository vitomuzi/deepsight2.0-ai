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


class img_label_scale_db(object):
	def __init__(self, cfg_path, cache_dir='data/cache'):

		with open(cfg_path, 'r') as f:
			data_cfg = yaml.load(f)
		f.close()
		print(cfg_path)

		self._img_dir = data_cfg['data_path']
		self._db_name = data_cfg['db_name']
		self._suffix = data_cfg['image_ext']
		self._class = data_cfg['class']

		self._cache_dir = cache_dir
		self._class_int = data_cfg['index']
		#self._class_to_ind = data_cfg['class_to_ind']
		self._normaldb = self._img_label_scale_db()

	@property
	def normaldb(self):
		return self._normaldb

	# generate _refinedb to refine detection result
	def _img_label_scale_db(self):
		# load cached data if it existed.
		cache_file = os.path.join(self._cache_dir, self._db_name+'_gt_db.pkl')
		if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				refinedb = pickle.load(fid)
			print('{} database loaded from {}'.format(self._db_name, cache_file))
			return refinedb

        # generate db used for refine result.
        # file list is generated from predict xml files
		file_name_itr = get_file_list(self._img_dir, file_suffix='.jpg')
		total_class = []
		total_img = []
		img_path_list = []
		for file_name in file_name_itr:
			img_file = os.path.join(self._img_dir, file_name+self._suffix)
			img = Image.open(img_file)
			current_class = self._class

			img_path_list.append([os.path.join(self._img_dir, file_name+self._suffix)])
			total_class.append(self._class_int)
			total_img.append(np.asarray(img))
            
		db_info = zip(total_class, img_path_list, total_img)
		refinedb = [dict(zip(('class', 'path', 'image'), item_info)) for item_info in db_info]
		with open(cache_file, 'wb') as fid:
			pickle.dump(refinedb, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote {} database to {}'.format(self._db_name, cache_file))
		return refinedb


if __name__ == '__main__':

	gt_dir = '/home/liuchang/code/classification_net_pytorch/data/own_data/test'

	db = refine_db(img_dir, gt_dir, data_dir, 'test_precision_pre', cache_dir='/home/liuchang/code/classification_net_pytorch/data/cache', img_suffix='.jpg')
	test = db.refinedb

	class_list = [item['class'] for item in test]
	print(len(class_list))
	print(sum(class_list))

	# save_dir = '/home/lc/code/NetKit/data/output/patch_0723'
	# db._save_bndboxes(save_dir)
