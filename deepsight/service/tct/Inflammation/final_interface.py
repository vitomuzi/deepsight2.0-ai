'''
重构版本
    因为原先用ImageFolder函数，导致测试时需要多加一层文件夹，这对于大检测数据集不可接受
    重新写imgload函数
    用
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    进行图片处理

'''
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
from dataloader import get_train_valid_loader
import random
import argparse
import math
import numpy as np

import densenet
import os

from tensorboardX import SummaryWriter


#===============================================================把每个样本文件夹列出来成一个列表#
def _get_case_dir(case_folder):   # 样本 direction
  case_dir = filter(os.path.isdir, [os.path.join(case_folder, f) for f in os.listdir(case_folder)])
  return [d for d in case_dir]

#===========================================对每个case文件夹(slide)统计输出
def Handle_Per_Slide(slide,net,gpu):
    
    #init
    num_mild_img = 0
    num_moderate_img = 0
    num_normal_img = 0

    #get img and transform
    file_list = os.listdir(slide)

    img_list = []
    for f in file_list:
        if '.jpg' in f:
            img_list.append(os.path.join(slide,f))


    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])

    for imgF in img_list:
        img = Image.open(imgF)
        img = transform(img)
        img = img.unsqueeze(0) #3D -> 4D
        print(img.size())

        if gpu:
            X = Variable(img).cuda()
            #y = Variable(label).cuda()
        else:
            X = Variable(img)
            #y = Variable(label)

        print(X.size())

        #========================================预测结果，传入参数：X 输出结果：y_predict
        predict = net(X)
        y_predict = F.softmax(predict)
        y_predict = torch.max(y_predict.data, dim=1)[1] 

        if y_predict[0] == 0:
            num_norm_img += 1
        elif y_predict[0] == 1:
            num_mild_img += 1
        elif y_predict[0] == 2:
            num_mode_img += 1
        
    print('normal: ',num_norm_img)
    print('mild: ',num_mild_img)
    print('moderate: ',num_mode_img)
        
       
    print('----------------------------------------------')


#==========================================MAIN
if __name__ == "__main__":
    device_ids = ['0','1']
    parser = argparse.ArgumentParser(description='Train a DenseNet.')
    parser.add_argument('--load', dest='load', help='load', type=bool, default=False)
    parser.add_argument('--gpu', dest='gpu', help='gpu', type=bool, default=False)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', type=int, default=1)
    args = parser.parse_args()
    batch_size = args.batch_size
    gpu = args.gpu
    load = args.load



    # criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.85, weight_decay=0.0005)

    net = densenet.DenseNet(
			num_classes=3,
			depth=46,
			growthRate=12,
			compressionRate=2,
			dropRate=0
			)
    
    if gpu:
        net = nn.DataParallel(net,device_ids=device_ids)
        net = net.cuda()
        

    if load:
        net.load_state_dict(torch.load('checkpoints/CP36.pth'))


    case_folder='/mnt/zhoum/large_scale_test/nanjinggulou_299'
    slides = _get_case_dir(case_folder)
    i = 0


    
    

    print('--------------Start Analyse----------------------')

    for slide in slides:
        print('slide ',i)
        Handle_Per_Slide(slide,net,gpu)
        i += 1





