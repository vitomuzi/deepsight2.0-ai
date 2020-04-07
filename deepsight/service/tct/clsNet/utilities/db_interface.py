from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as Data
from PIL import Image
import numpy as np
import math
import random

try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3

class db_interface(Data.Dataset):
	def __init__(self, input_db, image_size=128):
		self._input_db = input_db
		self._edge = image_size

	def __len__(self):
		return len(self._input_db)

	def __getitem__(self, index):
		item_info = self._input_db[index]
		item_class = item_info['class']
		item_data = item_info['image']

		img = Image.fromarray(item_data)
		scale_factor = min(self._edge/float(img.size[0]), self._edge/float(img.size[1]))
		img = img.resize([min(self._edge, math.ceil(img.size[0]*scale_factor)), min(self._edge, math.ceil(img.size[1]*scale_factor))], resample=Image.BILINEAR)

		data = np.asarray(img)
		data_padding = np.zeros((self._edge, self._edge, 3))
		w_head = math.floor((self._edge-img.size[0])/2.)
		h_head = math.floor((self._edge-img.size[1])/2.)
		data_padding[h_head:h_head+img.size[1], w_head:w_head+img.size[0], :] = data
		data = torch.from_numpy(np.rollaxis(data_padding, 2, 0)).float()

		return data, item_class

class sampler_interface(Data.sampler.Sampler):
	def __init__(self, data_size, random_seed=None):
		self._data_size = data_size
		self._index = [i for i in xrange(data_size)]
		random.shuffle(self._index)
		if random_seed!=None:
			random.seed(random_seed)

	def __len__(self):
		return self._data_size

	def __iter__(self):
		return iter(self._index)
