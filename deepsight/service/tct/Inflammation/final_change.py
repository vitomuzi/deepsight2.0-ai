# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import urllib
import urllib.error
import urllib.parse
import urllib.request

HOST = "https://deepsight.eyeblue.cn"

import numpy as np
import cv2
from scipy.misc import imread
import random

#import fasterRCNN._init_paths
#import fasterRCNN.interface as rcnn
#import fasterRCNN.utils.general as ut
from common.fasterRCNN import _init_paths
from common.fasterRCNN import interface as rcnn
from common.fasterRCNN.utils import general as ut

from tqdm import tqdm


#===============================================================把每个样本文件夹列出来成一个列表#
def _get_case_dir(case_folder):   # 样本 direction
  case_dir = filter(os.path.isdir, [os.path.join(case_folder, f) for f in os.listdir(case_folder)])
  return [os.path.join(case_folder, d) for d in case_dir]

#==============================================================从case文件夹中取出图片#
# 片子slide即case所对应的文件夹
def fetchImages(slide):
    file_itr = ut.get_file_list(img_dir=slide, img_suffix='.jpg')

    return [os.path.join(slide, f+'.jpg') for f in file_itr]

#==============================================================从num_boxes_array得到统计结果，得到结果dictionary#
# convert analysis array to upload content
#num_boxes_array is the stastic array---------------
def analysis_to_post(num_boxes_array):
    post_dict = {
        # 炎症  ["无", "轻度", "中度", "重度"]
        "inflammation":"无",
        # 细胞量 [">5000", "<5000"]
        "cellAmount":"<5000",
        #细胞项目->鳞状上皮 ["有", "未见"]
        "squamousEpithelium":"未见",
        # fixed 细胞项目->颈管细胞 ["有", "未见"]
        "endocervicalCell":"未见",
        # fixed 细胞项目->化生细胞 ["有", "未见"]
        "metaplasticCell":"未见",
        #病毒感染 ["HPV感染", "未见", "疱疹病毒感染"]
        "virusInfection":"未见",
        #微生物项目 ["阴道滴虫", "真菌，形态学上符合念珠菌属", "菌群变化，提示细菌性阴道病", "细菌，形态学上符合放线菌属", "细胞变化，符合单纯疱疹病毒感染", "无"]
        "microorganismProject":"无",
        #初筛鳞状上皮细胞分析 ["未见上皮内病变及恶性病变（NILM）", "不确定或者低级别病变", "高级别鳞状上皮病变"]
        "squamousEpitheliumResult":"未见上皮内病变及恶性病变（NILM）"
    }
    # class_cls = ('normal', 'abnormal', 'H-SIL', 'Candida', 'fungi', 'Herpesviruses')
    num_cell = np.sum(num_boxes_array)
    post_dict['cellAmount'] = '<5000' if num_cell<5000 else '>5000'
    post_dict['squamousEpithelium'] = '有' if np.sum(num_boxes_array[:3])>0 else '未见'
    if num_boxes_array[5] > 0:
        post_dict['virusInfection'] = '疱疹病毒感染'
    # 判断逻辑需要调整
    if num_boxes_array[4] > 0:
        post_dict['microorganismProject'] = '细菌，形态学上符合放线菌属'
    if num_boxes_array[3] > 0:
        post_dict['microorganismProject'] = '真菌，形态学上符合念珠菌属'
    # 判断逻辑需要优化
    if num_boxes_array[2] > 0:
        post_dict['squamousEpitheliumResult'] = '高级别鳞状上皮病变'
    elif num_boxes_array[1] > 0:
        post_dict['squamousEpitheliumResult'] = '不确定或者低级别病变'
    else:
        post_dict['squamousEpitheliumResult'] = '未见上皮内病变及恶性病变（NILM）'

    return post_dict




# ==========================================================================处理一张片子#
def handleSlide(slide, fasterRCNN, class_det):
    # 根据一个片子的编号，去下载用于分析的图片。
    # (因为目前并不是每个片子都对应有影像图，所以先统一用sample这个影像图作为所有片子的影像图)
    # 后面这个地方应该传 slideNo
    file_list = fetchImages(slide)

    num_all_img = 0
    num_abnormal_img = 0

    # the uploaded content is sampled from the contentArray
    # type_dict = {cls name of net output: upload cls name}
    type_dict = {'normal':'SC', 'abnormal':'LSIL', 'H-SIL':'H_SIL', 'Candida':'NILM', 'fungi':'NILM', 'Herpesviruses':'NILM'}
    key_list = [key for key in type_dict]
    # init contentArray
    contentArray = {}
    for key in key_list:
        contentArray[key] = []
    # record the num of boxes for each array
    #num_boxes_array = np.zeros(len(class_cls))
    num_boxes_array = np.zeros(len(class_det))

    for image_path in tqdm(file_list):
        im_in = np.array(imread(image_path))
        [dir_name, file_name]=os.path.split(image_path)
        #objs_info = rcnn.inference(im_in, fasterRCNN, class_det, cfg_file='fasterRCNN/cfgs/vgg16.yml', cuda=True, cfg_list=cfg_list)
        objs_info = rcnn.inference(im_in, fasterRCNN, class_det, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list)
       # num_boxes_array = num_boxes_array + num_boxes_array_img

        ##统计每张视野图是否有异常视野
        num_all_img += 1
        for class_name in objs_info['classes_name']:
            if class_name == 'abnormal':
                num_abnormal_img += 1
                break

    #print(num_boxes_array)
    print('inflammation : ',num_abnormal_img, '  All : ',num_all_img)
    ##确定阈值的部分
    # final_label = 'normal'
    # if num_abnormal_img>num_all_img*0.5:
    #     final_label = 'abnormal'
    # print(final_label)



#========================================================================MAIN#
if __name__ == "__main__":
    # load models
    #class_det = ('__background__', '2')
    class_det = ('__background__', 'abnormal')
    # faster RCNN
    header='densenet'
    model_path = 'checkpoints/CP11.pth'
    #cfg_file = 'fasterRCNN/cfgs/{}.yml'.format(header)
    #cfg_list = ['ANCHOR_SCALES', '[16, 32, 64, 128]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    fasterRCNN = rcnn.load_model(model_path, class_det, header=header, cuda=True)
    print('faster RCNN loaded')


    # 拉取所有尚未分析过的片子
    #case_folder='/home/zhouming/data/cc/train/case'
    case_folder='/inflammation/MIild'
    slides = _get_case_dir(case_folder)

    # 对每张片子进行处理
    for slide in slides:
        handleSlide(slide, fasterRCNN, class_det)

    # 完成
    print("-------------Done-------------")
