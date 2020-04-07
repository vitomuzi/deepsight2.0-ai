# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil
import urllib
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
import cv2
from scipy.misc import imread
import random

import _init_paths
import interface as interface
import data.utils as utls

from tqdm import tqdm

# get file name iterator
def get_file_list(img_dir, img_suffix='.jpg'):
    f_all_itr = (f for f in os.listdir(img_dir))
    f_itr = filter(lambda f:f.endswith(img_suffix), sorted(f_all_itr))
    f_itr = map(lambda f:f[:-len(img_suffix)], f_itr)
    return f_itr

def _get_case_dir(case_folder):
  case_dir = filter(os.path.isdir, [os.path.join(case_folder, f) for f in os.listdir(case_folder)])
  return [os.path.join(case_folder, d) for d in case_dir]


# 片子slide即case所对应的文件夹
def fetchImages(slide):
    file_itr = get_file_list(img_dir=slide, img_suffix='.jpg')

    return [os.path.join(slide, f+'.jpg') for f in file_itr]



# 处理一张片子
def handleSlide(slide, clsNet, class_cls, annt_info_outdir):
    # 根据一个片子的编号，去下载用于分析的图片。
    # (因为目前并不是每个片子都对应有影像图，所以先统一用sample这个影像图作为所有片子的影像图)
    # 后面这个地方应该传 slideNo
    file_list = fetchImages(slide)

    # the uploaded content is sampled from the contentArray
    # type_dict = {cls name of net output: upload cls name}
    # type_dict = {'normal':'SC', 'abnormal':'LSIL', 'H-SIL':'H_SIL', 'Candida':'NILM', 'fungi':'NILM', 'Herpesviruses':'NILM'}
    # key_list = [key for key in type_dict]
    
    # record the num of boxes for each array

    num = 0
    for image_path in tqdm(file_list):
        im_in = np.array(imread(image_path))
        img = Image.fromarray(im_in)
        [dir_name, file_name]=os.path.split(image_path)

        annt_path = image_path.split('.')[0]+'.xml'
        annt_file,tree = utls.load_pre_annt_file(annt_path)

        boxes, classes, patches, total_prob = interface.get_testImg(im_in, annt_file,class_cls)
        
        predict_class, predict_result, result_flag = interface.interface(patches, total_prob, net, class_cls, cuda=True)

        predict_class_list = []
        predict_class_list.append(predict_class)
        print(file_name+' '+str(predict_class_list[0])+' '+str(predict_result)+' '+ str(result_flag))

        num += predict_class
        np.savetxt(annt_info_outdir+ '/' + file_name + '.txt', predict_class_list, fmt = '%d')
    
    print(slide + '  ' +str(num))
    # analysis_post_dict = analysis_to_post(num_boxes_array)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference on local')
    parser.add_argument('--num_fold', help='the num of fold for test', default=1, type=int)
    parser.add_argument('--index', help='choose a part of f_itr', default=0, type=int)
    args = parser.parse_args()
    

    # cls net
    #class_cls = ('normal', 'abnormal', 'H-SIL', 'Candida', 'fungi', 'Herpesviruses')
    class_cls = ('normal', 'abnormal')
    net_path = 'data/models/CP281.pth'

    net = interface.load_model(net_path, class_cls, cuda=True)
    print('DenseNet loaded')

    # 拉取所有尚未分析过的片子
    case_folder = '/root/zhoum/CC/densenet_step3_v2/testdata'
    slides = _get_case_dir(case_folder)

    #输出预测的标注结果文件夹
    annt_info_outdir = '/root/zhoum/CC/densenet_step3_v2/testoutput/0605_2'


    # 对每片子进行处理
    for slideIndex , slide in enumerate(slides):
        if slideIndex % args.num_fold ==  args.index:
            print(slide+'out')
            handleSlide(slide, net, class_cls,annt_info_outdir)

    # 完成
    print("-------------Done-------------"+str(args.index))
