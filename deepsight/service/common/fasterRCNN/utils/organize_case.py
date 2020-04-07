import numpy as np
import math
import os
import shutil
from PIL import Image

import general as ut

if __name__ == '__main__':
	# for faster rcnn================
	_dir = '/home/zhouming/data/cc/test_multilabel/'
	out_dir = '/home/zhouming/data/cc/dir_train_for_case/H-SIL_dir'
	file_itr = ut.get_file_list(img_dir=_dir, img_suffix='.jpg')
	
	for file_name in file_itr:
		case_name = file_name.split('_')[0][:-8]
		#case_name = file_name.split('_')[0][:-10]
		if not os.path.exists(os.path.join(out_dir, case_name)):
			os.makedirs(os.path.join(out_dir, case_name))
		shutil.copy(os.path.join(_dir, file_name+'.jpg'), os.path.join(out_dir, case_name))
		try:
			shutil.copy(os.path.join(_dir, file_name+'.xml'), os.path.join(out_dir, case_name))
		except Exception as e:
			print(e)
		print(file_name)
		print(case_name)