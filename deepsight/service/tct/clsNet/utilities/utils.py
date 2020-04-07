import numpy as np
import SimpleITK as sitk
import os
import random
import math
import xml.etree.ElementTree as ET

# get file name iterator. (no suffix, no file path)
def get_file_list(file_dir, file_suffix='.jpg'):
	f_all_itr = (f for f in os.listdir(file_dir))
	f_itr = filter(lambda f:f.endswith(file_suffix), sorted(f_all_itr))
	f_itr = map(lambda f:f[:-len(file_suffix)], f_itr)
	return f_itr

#==================================================
# split data set into validation set and training set
def data_set_split(data, val_percent=0.1, seed_random=None):
	data = list(data)
	num_data = len(data)
	num_val = int(num_data * val_percent)
	if seed_random!=None:
		random.seed(seed_random)
	random.shuffle(data)
	return {'train': data[:-num_val], 'val': data[-num_val:]}

# construct a batch data generator
def batch(data, size_batch):
	batch = [];
	for i, sbj in enumerate(data):
		batch.append(sbj)
		if (i+1) % size_batch == 0:
			yield batch
			batch = []

	if len(batch) > 0:
		yield batch

#==================================================
# load data into a list:
def load_img(data_dir, path_itr, suffix):
	data_list = []
	for data_path in path_itr:
		data_path = data_dir + '/' + data_path + suffix
		img = sitk.ReadImage(data_path)
		data_list.append(sitk.GetArrayFromImage(img))
	return data_list

def extract_slice(img_itr, index_list):
	slice_list = []
	index_list = list(index_list)
	for i, img in enumerate(img_itr):
		slice_list.append(img[index_list[i],:,:])
	return slice_list

#==================================================
# load annotation file and return objects information
def load_annt_file(annt_path):
	tree = ET.parse(annt_path)
	objs = tree.findall('object')
	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	classes = []
	ishards = np.empty(shape=[0], dtype=np.int32)
	for obj in objs:
		bbox = obj.find('bndbox')
		x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text)>0 else 0
		y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text)>0 else 0
		x2 = float(bbox.find('xmax').text) - 1
		y2 = float(bbox.find('ymax').text) - 1
		if x1 >= x2 or y1>=y2:
		    continue

		diffc = obj.find('difficult')
		difficult = 0 if diffc == None else int(diffc.text)
		ishards = np.append(ishards, difficult).astype(np.int32)

		class_name = obj.find('name').text.lower().strip()
		# # the following line is used for dirty data
		# class_name = 'abnormal'

		boxes = np.append(boxes, np.expand_dims([x1, y1, x2, y2], axis=0).astype(np.uint16), axis=0)
		classes.append(class_name)

	return {'boxes': boxes,
    		'classes_name': classes,
    		'gt_ishard': ishards}, tree

#==================================================
# load annotation file and return objects information
def load_pre_annt_file(annt_path):
	tree = ET.parse(annt_path)
	objs = tree.findall('object')
	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	classes = []
	ishards = np.empty(shape=[0], dtype=np.int32)
	probablity_s = np.empty(shape=[0], dtype=np.float16)
	for obj in objs:
		bbox = obj.find('bndbox')
		x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text)>0 else 0
		y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text)>0 else 0
		x2 = float(bbox.find('xmax').text) - 1
		y2 = float(bbox.find('ymax').text) - 1
		if x1 >= x2 or y1>=y2:
		    continue

		diffc = obj.find('difficult')
		difficult = 0 if diffc == None else int(diffc.text)
		ishards = np.append(ishards, difficult).astype(np.int32)
		#get probablity
		proba = obj.find('probablity')
		probablity = float(proba.text)
		probablity_s = np.append(probablity_s, probablity).astype(np.float16)

		class_name = obj.find('name').text.lower().strip()
		# # the following line is used for dirty data
		# class_name = 'abnormal'

		boxes = np.append(boxes, np.expand_dims([x1, y1, x2, y2], axis=0).astype(np.uint16), axis=0)
		classes.append(class_name)

	return {'boxes': boxes,
    		'classes_name': classes,
    		'gt_ishard': ishards,
    		'gt_probablity': probablity_s}, tree

# save annotation file for cropped images.
def save_annt_file(file_path, objs_info, template_tree=None):
	if template_tree == None:
		root = ET.Element('annotation')
		folder = ET.SubElement(root, 'folder')
		folder.text = 'undefined'
		filename = ET.SubElement(root, 'filename')
		filename.text = 'undefined'
		path = ET.SubElement(root, 'path')
		path.text = 'undefined'
		source = ET.SubElement(root, 'source')
		database = ET.SubElement(source, 'database')
		database.text = 'unknown'
		size = ET.SubElement(root, 'size')
		width = ET.SubElement(size, 'width')
		height = ET.SubElement(size, 'height')
		depth = ET.SubElement(size, 'depth')
		width.text = str(0)
		height.text = str(0)
		depth.text = str(0)
		segmented = ET.SubElement(root, 'segmented')
		segmented.text = str(0)
	else:
		root = template_tree.getroot()
		objs = template_tree.findall('object')
		for obj in objs:
			root.remove(obj)

	for i in range(len(objs_info['classes_name'])):
		obj = ET.SubElement(root, 'object')
		name = ET.SubElement(obj, 'name')
		name.text = objs_info['classes_name'][i]
		pose = ET.SubElement(obj, 'pose')
		pose.text = 'Unspecified'
		truncated = ET.SubElement(obj, 'truncated')
		truncated.text = str(0)
		difficult = ET.SubElement(obj, 'difficult')
		difficult.text = str(objs_info['gt_ishard'][i])
		probablity = ET.SubElement(obj, 'probablity')
		probablity.text = str("%.3f" %objs_info['gt_probablity'][i])
		bndbox = ET.SubElement(obj, 'bndbox')
		xmin = ET.SubElement(bndbox, 'xmin')
		ymin = ET.SubElement(bndbox, 'ymin')
		xmax = ET.SubElement(bndbox, 'xmax')
		ymax = ET.SubElement(bndbox, 'ymax')
		xmin.text = str(int(np.mat(objs_info['boxes'])[i,0]))
		ymin.text = str(int(np.mat(objs_info['boxes'])[i,1]))
		xmax.text = str(int(np.mat(objs_info['boxes'])[i,2]))
		ymax.text = str(int(np.mat(objs_info['boxes'])[i,3]))
	
	tree = ET.ElementTree(root)
	tree.write(file_path, encoding='UTF-8')