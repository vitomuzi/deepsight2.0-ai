#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: load.py
# Project: deepsight-back
# Created Date: 十二月 26,2019 11:32
# Author: yanlin
# Description: 加载fasterRCNN模型
# -----
# Last Modified: 2019/12/26 11:32
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
import os

from . import _init_paths

from . import interface as rcnn

#faster RCNN load
def faster_rcnnload(class_det, header, cfg_list):
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

    cfg_file = os.path.join(BASE_DIR,'common/fasterRCNN/cfgs/{}.yml'.format(header))

    model_path = os.path.join(BASE_DIR,'common/fasterRCNN/data/models/res101/own_data/faster_rcnn_abnormal_ah128_bs16_1_20_1470.pth')

    fasterRCNN = rcnn.load_model(model_path, class_det, header=header, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list)

    return fasterRCNN, cfg_file

