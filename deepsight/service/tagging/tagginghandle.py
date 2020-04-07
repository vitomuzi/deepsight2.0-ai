#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: tagginghandle.py
# Project: deepsight-back
# Created Date: 十二月 25,2019 12:49
# Author: yanlin
# Description: 标记模型分析
# -----
# Last Modified: 2019/12/25 12:49
# Modified By: yanlin
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import numpy as np
from . import _init_paths
from deepsight.db import db
from scipy.misc import imread
from django.conf import settings
import xml.etree.ElementTree as ET
from deepsight.service.common.fasterRCNN import interface as rcnn
from deepsight.formatter.json.create_json import CreateTaggingJson


#产生新的xml文件名称
def _get_targetIMG_file_path(filename):
    f_name,fe_name = os.path.splitext(filename)
    return f_name+'.jpg'



def get_file_list(img_dir, img_suffix='.jpg'):
    f_all_itr={}
    for i_name in os.listdir(img_dir):
        m = os.path.join(img_dir,i_name)
        if os.path.isfile(m):
            f_all_itr[i_name]=os.path.getsize(m)
    sort_result = sorted(f_all_itr.items(),key = lambda asd:asd[1],reverse=True)

    f_itr = []
    for item in sort_result:
        if item[0].endswith(img_suffix):
            f_itr.append(item[0])
    f_itr = map(lambda f:f[:-len(img_suffix)], f_itr)

    return f_itr


# 片子slide即case所对应的文件夹
def fetchImages(slide):
    file_itr = get_file_list(img_dir=slide, img_suffix='.jpg')

    return [os.path.join(slide, f+'.jpg') for f in file_itr]


'''
function : update_picture_analysis 批量更新图片分析的信息到数据库
conn : 数据库句柄
number : 该图片的编号
all_lists : 需要插入字段的列表
'''
def update_picture_analysis(conn, number, all_lists):
    for all_list in all_lists:
        db.DbOperation(conn=conn, sample_id=number, start_time=all_list['start_time'], finish_time=all_list['finish_time'], duration=all_list['duration'], is_completed=1, result=all_list['result'], image_name=all_list['image_name']).db_update_results()


# 处理一张片子
def handleSlide(conn, slide, fasterRCNN, class_det, cfg_file, cfg_list,annt_info_outdir, number, check_num = 200):
    # 根据一个片子的编号，去下载用于分析的图片。
    # (因为目前并不是每个片子都对应有影像图，所以先统一用sample这个影像图作为所有片子的影像图)
    # 后面这个地方应该传 slideNo
    file_list = fetchImages(slide)
    #最大考察的视野图个数
    if len(file_list)>check_num:
        file_list = file_list[:check_num:1]
    # the uploaded content is sampled from the contentArray
    # type_dict = {cls name of net output: upload cls name}
    type_dict = {'normal':'SC', 'abnormal':'LSIL', 'H-SIL':'H_SIL', 'Candida':'NILM', 'fungi':'NILM', 'Herpesviruses':'NILM'}
    key_list = [key for key in type_dict]
    # init contentArray
    contentArray = {}
    for key in key_list:
        contentArray[key] = []
    #print(contentArray)
    #存贮每个图片分析的坐标
    content_data = []
    # 创建一个空列表，收集这个样本所有的图片信息
    picture_lists = []


    for image_path in (file_list):

        im_in = np.array(imread(image_path))
        [dir_name, file_name]=os.path.split(image_path)
        #数据库要保存的时间
        save_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #开始分析的时间
        start_time = time.time()
        objs_info = rcnn.inference(im_in, fasterRCNN, class_det, cfg_file= settings.BASE_DIR+'/deepsight/service/common/fasterRCNN/cfgs/res101.yml', cuda=True, cfg_list=cfg_list)
        #分析结束的时间
        end_time = time.time()
        #用于需要保存分析结束的时间
        save_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        total_time = int(end_time-start_time)
        #分析出来的框  json串存在数据库中
        tmp_boxes = objs_info['boxes'].tolist()
        tmp_boxes = json.dumps(tmp_boxes)

        picture_name = _get_targetIMG_file_path(file_name)
        # 创建一个空字典，收集每一个图片的图片信息
        picture_dict = {}
        #更新数据库
        picture_dict['start_time'] = save_start_time
        picture_dict['finish_time'] = save_end_time
        picture_dict['duration'] = total_time
        picture_dict['result'] = tmp_boxes
        picture_dict['image_name'] = picture_name
        picture_lists.append(picture_dict)
        #print(objs_info)
        tmp_dict = {
            "imageUrl": "",
            "rects":[]
        }
        #组装jpg的地址
        image_url = os.path.join(settings.UPLOAD_URL,number)
        image_url = os.path.join(image_url,picture_name)
        tmp_dict["imageUrl"] = image_url
        tmp_dict["rects"].append(objs_info['boxes'].tolist())
        content_data.append(tmp_dict)
    #调用更新数据库操作
    update_picture_analysis(conn, number, picture_lists)
    tagging_json = CreateTaggingJson(number, content_data)
    result_json = tagging_json.create_tagging_json()

    return result_json
