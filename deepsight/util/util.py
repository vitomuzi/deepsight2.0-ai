#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: util.py
# Project: deepsight-back
# Created Date: 一月 09,2020 15:30
# Author: yanlin
# Description:  公共函数
# -----
# Last Modified: 2020/1/9 15:30
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
import shutil
import logging
from django.conf import settings
logger = logging.getLogger(settings.ENVTYPE)

'''
function : check_folder 检查文件夹是否存在，不存在创建
filepaht : 文件夹路径
'''
def check_folder(filepath):
    # 判断地址是否存在，不存在创建
    if os.path.exists(filepath):
        # 存在先清空再创建
        logger.debug('文件存在，清空文件后再创建文件夹  ' + filepath)
        shutil.rmtree(filepath)
        os.makedirs(filepath)
    else:
        logger.debug('创建文件夹  ' + filepath)
        os.makedirs(filepath)