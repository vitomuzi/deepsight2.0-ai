# import numpy as np
# import os
# import pickle
# import torch
# import torch.nn.functional as F
# import torch.utils.data as data

# from torch.autograd import Variable

# INPUT_DIM = 224
# MAX_PIXEL_VAL = 255
# MEAN = 58.09
# STDDEV = 49.73

# class Dataset(data.Dataset):
#     def __init__(self, datadirs, diagnosis, use_gpu):
#         super().__init__()
#         self.use_gpu = use_gpu

#         label_dict = {}
#         self.paths = []

#         for i, line in enumerate(open('metadata.csv').readlines()):
#             if i == 0:
#                 continue
#             line = line.strip().split(',')
#             path = line[10]
#             label = line[2]
#             label_dict[path] = int(int(label) > diagnosis)

#         for dir in datadirs:
#             for file in os.listdir(dir):
#                 self.paths.append(dir+'/'+file)

#         self.labels = [label_dict[path[6:]] for path in self.paths]

#         neg_weight = np.mean(self.labels)
#         self.weights = [neg_weight, 1 - neg_weight]

#     def weighted_loss(self, prediction, target):
#         weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
#         weights_tensor = torch.FloatTensor(weights_npy)
#         if self.use_gpu:
#             weights_tensor = weights_tensor.cuda()
#         loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
#         return loss

#     def __getitem__(self, index):
#         path = self.paths[index]
#         with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
#             vol = pickle.load(file_handler).astype(np.int32)

#         # crop middle
#         pad = int((vol.shape[2] - INPUT_DIM)/2)
#         vol = vol[:,pad:-pad,pad:-pad]
        
#         # standardize
#         vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

#         # normalize
#         vol = (vol - MEAN) / STDDEV
        
#         # convert to RGB
#         vol = np.stack((vol,)*3, axis=1)

#         vol_tensor = torch.FloatTensor(vol)
#         label_tensor = torch.FloatTensor([self.labels[index]])

#         return vol_tensor, label_tensor

#     def __len__(self):
#         return len(self.paths)

# def load_data(diagnosis, use_gpu=False):
#     train_dirs = ['vol08','vol04','vol03','vol09','vol06','vol07']
#     valid_dirs = ['vol10','vol05']
#     test_dirs = ['vol01','vol02']
    
#     train_dataset = Dataset(train_dirs, diagnosis, use_gpu)
#     valid_dataset = Dataset(valid_dirs, diagnosis, use_gpu)
#     test_dataset = Dataset(test_dirs, diagnosis, use_gpu)

#     train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
#     valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
#     test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

#     return train_loader, valid_loader, test_loader

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import *


class CustomDataset(Dataset):
    def __init__(self, config, name='train', shuffle=True):
        self.root = config['DataRoot']
        self.config = config
        self.name = name
        if self.name == 'train':
            file_path = os.path.join(self.root, 'imageSets', self.config['Train']['trainSet'])
        elif self.name == 'val':
            file_path = os.path.join(self.root, 'imageSets', self.config['Train']['valSet'])
        # elif self.name == 'test':
        #     file_path = os.path.join(self.root, 'imageSets', self.config['Test']['testSet'])
        with open(file_path, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        self.nSamples = len(self.lines)
        self.length = self.config['length']

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        imgpath = self.lines[index].rstrip()
        if self.name == 'train':
            img, label_index = self.load_train_data(imgpath, self.config)
            # img = torch.from_numpy(img).float()
#            label_index = torch.LongTensor([label_index])
        elif self.name == 'val' or self.name == 'test':
            img, label_index = self.load_train_data(imgpath, self.config)
 #           label_index = torch.LongTensor([label_index])

        return (img ,label_index)


    def load_train_data(self, imgpath, config):
        classes = self.config['Data_CLASSES']
        im_h = self.config['img_size'][0]
        im_w = self.config['img_size'][1]
        mean = self.config['Means']
        std = self.config['Stds']

        seq = np.zeros((5, im_h, im_w, 3), dtype=np.float32)

        img_path = imgpath.split("   ")[0]
        label_path = imgpath.split("   ")[1]
        label_index = int(classes.index(label_path))

        img_file = os.path.join(self.config['img_dir'], img_path+self.config['image_ext'])
        annt_file = os.path.join(self.config['annt_dir'], img_path+'.xml')

        try:
            img = Image.open(img_file)
            boxes, patches, probablity_s = _get_box_info(annt_file, img)
        except ValueError as e:
            print(e)
        except OSError as e:
            print(e)
                

        len_db = len(probablity_s)
        if len_db==0:
            print(img_path)

        if len_db<5:
            probablity_s_ext = probablity_s+probablity_s * math.floor(5/len_db+1)
            patches_ext = patches+patches * math.floor(5/len_db+1)
        else:
            probablity_s_ext = probablity_s
            patches_ext = patches

        pro_index = np.argsort(probablity_s_ext)
        # print(len(db_current_ext))
        # print(db_probablity_s)
        # print(len(pro_index))
        #按照probablity的大小取得img 顺序排列
        for img_index in range(1,6):
            seq_index = img_index - 1
            # 顺序的次序 pro_index[-img_index]
            item_data = patches_ext[pro_index[-img_index]]
            #img = np.array(item_data)

            seq[seq_index,:,:,0] = (item_data[:,:,0]/255.-mean[0])/std[0]
            seq[seq_index,:,:,1] = (item_data[:,:,1]/255.-mean[1])/std[1]
            seq[seq_index,:,:,2] = (item_data[:,:,2]/255.-mean[2])/std[2]


        data = torch.from_numpy(np.rollaxis(seq, 3, 1)).float()


        return data ,label_index


def _get_box_info( annt_file, img):
    if not(os.path.exists(annt_file)):
        print(''''{} doesn\'t exist.'''.format(annt_file))
        return [], [], []
    else:
        annt_info, elemet_tree = load_pre_annt_file(annt_file)
    boxes = []
    patch = []
    total_probality = []
    if len(annt_info['classes_name']) == 0:
            # return empty list if annt file is empty
        return boxes, patch, total_probality

    for idx, box_annt in enumerate(annt_info['boxes']):
        box_item, patch_item = _get_img(box_annt, img)
        boxes.extend(box_item)
        patch.extend(patch_item)
        total_probality.append(annt_info['gt_probablity'][idx])

        # return boxes, classes, patch
    return boxes, patch, total_probality

def _get_img( box, img ):
    xmin, ymin, xmax, ymax = box
    x_center = (xmin+xmax)/2
    y_center = (ymin+ymax)/2
    x_center = 112 if x_center<112 else x_center
    y_center = 112 if y_center<112 else y_center
    x_center = 912 if x_center>912 else x_center
    y_center = 912 if y_center>912 else y_center

    xmin = x_center - 112
    xmax = x_center + 112
    ymin = y_center - 112
    ymax = y_center + 112
    box = [xmin, ymin, xmax, ymax]

    return [box], [np.asarray(img.crop(box))]

def load_data(diagnosis, use_gpu=False):
    return train_loader, valid_loader, test_loader


