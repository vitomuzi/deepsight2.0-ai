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
# load prediced annotation file and return objects information
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
		probabl = obj.find('probablity')
		probablity = float(probabl.text)
		probablity_s = np.append(probablity_s, probablity).astype(np.float16)

		class_name = obj.find('name').text.lower().strip()
		# # the following line is used for dirty data
		# class_name = 'abnormal'

		boxes = np.append(boxes, np.expand_dims([x1, y1, x2, y2], axis=0).astype(np.uint16), axis=0)
		classes.append(class_name)

	return {'boxes': boxes,
    		'classes_name': classes,
    		'gt_ishard': ishards,
    		'gt_probablity':probablity_s}, tree

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

def getCase_CellImg(img_path,annt_path,cell_size,num_img):
	file_itr = get_file_list(file_dir=img_path, file_suffix='jpg')
	total_probality = []
	for file_name in file_itr:
		try:
			im = imread(os.path.join(img_path, file_name+'.jpg'))
			annt_file = os.path.join(annt_path, file_name+'.xml')
			annt_info, _ = load_pre_annt_file()
			patches = _get_box_img()
		except ValueError as e:
			print(e)
			return
		except OSError as e:
			print(e)
			return

		probablity_s = annt_info['gt_probablity']
		total_probality.extend(probablity_s)


		pro_index = np.argsort(probablity_s)

		for img_index in xrange(1,num_img+1):
			# pro_index[-img_index]  从大到小的index
			probablity_s[pro_index[-img_index]]



def getAttention(annt_path):
	tree = ET.parse(annt_path)
	objs = tree.findall('object')
	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	classes = []
	ishards = np.empty(shape=[0], dtype=np.int32)
	attention = np.zeros((1024,1024), dtype = np.int)

	for obj in objs:
		bbox = obj.find('bndbox')
		x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text)>0 else 0
		y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text)>0 else 0
		x2 = float(bbox.find('xmax').text) - 1
		y2 = float(bbox.find('ymax').text) - 1
		if x1 >= x2 or y1>=y2:
		    continue

		for row in xrange(x1, x2+1):
			for column in xrange(y1, y2+1):
				attention[row][column] = 255
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
    		'gt_ishard': ishards}, tree , attention

def _aug_img_2d(img_path, annt_path, out_dir, own_data_classes, aug_factor=60, trim_size=512, random_seed=False, overlap_th=0.4, ratio_th=2.5):
	get_sub_path = lambda sub_index, ori_path, out_dir:os.path.join(out_dir, os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1])

	try:
		im = imread(img_path)
		annt_info, template_tree = tools.load_annt_file(annt_path)
	except ValueError as e:
		print(e)
		return
	except OSError as e:
		print(e)
		return

	if len(annt_info['gt_classes_name'])==0:
		print('no bndboxes in this file.')
		return

	# random crop
	if random_seed:
		random.seed(random_seed)
	gt_boxes_ori = torch.from_numpy(annt_info['boxes'].astype(np.float))
	class_to_ind_dict = dict(zip(own_data_classes, range(len(own_data_classes))))
	ind_to_class_dict = dict(zip(range(len(own_data_classes)), own_data_classes))
	gt_classes_np = np.array([class_to_ind_dict[item] for item in annt_info['gt_classes_name']])
	gt_classes_ori = torch.from_numpy(gt_classes_np)

	gt_area = (gt_boxes_ori[:,2]-gt_boxes_ori[:,0])*(gt_boxes_ori[:,3]-gt_boxes_ori[:,1])
	gt_ratio = (gt_boxes_ori[:,2]-gt_boxes_ori[:,0])/(gt_boxes_ori[:,3]-gt_boxes_ori[:,1])
	gt_ratio[gt_ratio>ratio_th-1e-5]=0
	gt_ratio[gt_ratio<1.0/ratio_th+1e-5]=0

	if torch.sum(gt_ratio) < 1e-5:
		return

	# crop img
	height, width = im.shape[0], im.shape[1]
	min_x_bnd = max(0, torch.min(gt_boxes_ori[:,0])-trim_size)
	min_y_bnd = max(0, torch.min(gt_boxes_ori[:,1])-trim_size)
	max_x_bnd = min(width-trim_size, torch.max(gt_boxes_ori[:,2]))
	max_y_bnd = min(height-trim_size, torch.max(gt_boxes_ori[:,3]))

	sub_index = 0
	for sub_index_x in range(aug_factor):
		sub_img_path = get_sub_path(sub_index, img_path, out_dir)
		sub_annt_path = get_sub_path(sub_index, annt_path, out_dir)

		while True:
			gt_boxes = gt_boxes_ori.clone()
			gt_classes = gt_classes_ori.clone()
			x_min_img, y_min_img = random.randint(min_x_bnd, max_x_bnd), random.randint(min_y_bnd, max_y_bnd)
			gt_boxes[:,::2] = gt_boxes[:,::2] - x_min_img
			gt_boxes[:,1::2] = gt_boxes[:,1::2] - y_min_img
			gt_boxes.clamp_(0, trim_size - 1)
			overlap = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])/gt_area
			ratio = (gt_boxes[:,2]-gt_boxes[:,0])/(gt_boxes[:,3]-gt_boxes[:,1])
			boxes_width = gt_boxes[:,2]-gt_boxes[:,0]
			boxes_height = gt_boxes[:,3]-gt_boxes[:,1]
			# remove bndbox
			not_keep = (overlap < overlap_th) | (ratio > ratio_th) | (ratio < (1/ratio_th)) | (boxes_width<5) | (boxes_height<5)
			keep = torch.nonzero(not_keep == 0).view(-1)
			if keep.numel() == 0:
				continue
		
			gt_boxes = gt_boxes[keep]
			gt_classes = gt_classes[keep]
			annt_info['boxes'] = gt_boxes.numpy().astype(np.uint16)
			annt_info['gt_classes_name'] = [ind_to_class_dict[item] for item in list(gt_classes.numpy().astype(np.int32))]
			annt_info['ishards'] = np.zeros(len(annt_info['gt_classes_name'])).astype(np.int32)
	
			img = im[
				y_min_img:y_min_img+trim_size, 
				x_min_img:x_min_img+trim_size,
				:]
			break

		pil_img = Image.fromarray(img)

		tools.save_annt_file(sub_annt_path, annt_info, template_tree)
		pil_img.save(sub_img_path)
		sub_index += 1