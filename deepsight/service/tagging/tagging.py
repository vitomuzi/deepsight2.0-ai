#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: tagging.py
# Project: deepsight-back
# Created Date: 十二月 25,2019 12:49
# Author: yanlin
# Description: 标记模型分析
# -----
# Last Modified: 2020/01/08 14:49
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
import logging
from . import tagginghandle
from deepsight.util import util
from django.conf import settings
from deepsight.service.common.fasterRCNN import interface as rcnn
from deepsight.service.common.download_taggingfile import DownLoadTagging

logger = logging.getLogger(settings.ENVTYPE)


'''
class : Tagging 处理
number : 样本编号
urls : 要下载的4x3k 的10张图片的url
'''
class Tagging:
    def __init__(self,conn, number, urls):
        self.conn = conn
        self.number = number
        self.urls = urls

    '''
        method : tagging 加载分析模型和调用分析处理
    '''
    def tagging(self):
        # 加载4x3k 图片下载的的处理
        download_pic = DownLoadTagging(self.conn, self.number, self.urls)
        result_pic = download_pic.down_load_tagging()
        if result_pic == 500:
            logger.warning('tagging.py 中处理图片下载失败')
        class_det = ('__background__', '2')
        # faster RCNN
        header='res101'
        model_path = os.path.join(settings.BASE_DIR,'deepsight/service/tagging/model/faster_rcnn_sc_ah128_bs16_1_20_726.pth')
        cfg_file = os.path.join(settings.BASE_DIR,'deepsight/service/common/fasterRCNN/cfgs/{}.yml'.format(header))
        cfg_list = ['ANCHOR_SCALES', '[16, 32, 64, 128]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        #加载 fasterRCNN模型
        logger.debug('fasterRCNN model starts loading....')
        fasterRCNN = rcnn.load_model(model_path, class_det, header=header, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list)
        logger.debug('fasterRCNN model loaded....')
        # 拉取所有尚未分析过的片子
        case_folder=os.path.join(settings.TAGGING_IMAGE, self.number)
        #输出预测的标注结果文件夹
        annt_info_outdir = os.path.join(settings.TAGGING_IMAGE, self.number)

        # 对每张片子进行处理
        try:
            result_json = tagginghandle.handleSlide(self.conn, case_folder, fasterRCNN, class_det, cfg_file, cfg_list,annt_info_outdir, self.number, check_num = 200)
            return result_json
        except Exception as e:
            logger.warning("tagging.py 文件中处理失败，错误内容如下：")
            logger.warning(e)

