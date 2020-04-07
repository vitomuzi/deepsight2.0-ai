#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: download.py
# Project: deepsight-back
# Created Date: 一月 07,2020 09:09
# Author: yanlin
# Description: 下载传过来的单个的文件
# -----
# Last Modified: 2020/1/7 9:09
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
import logging
import urllib.request
from django.conf import settings

logger = logging.getLogger(settings.ENVTYPE)

'''
class : download  下载功能
save_folder : 下载路径保存的地址
url : url 例如 "http: //xxxx/1.jpg"  或者 "http: //xxxx/1.xml"
'''
class Download:
    def __init__(self, save_folder, url):
        self.save_folder = save_folder
        self.url = url

    def download(self):
        try:
            #下载图片
            logger.debug("download url is : %s "  % self.url )
            response = urllib.request.urlopen(self.url)
            img = response.read()
            # 获取下载图片的名字
            picture_name = os.path.basename(self.url)
            logger.debug(picture_name)
            # 组装要保存的地址路径
            save_picture_path = os.path.join(self.save_folder, '%s' % picture_name)
            with open(save_picture_path, 'wb') as code:
                code.write(img)
                logger.debug("save path :" + save_picture_path)
            return 200, picture_name
        except Exception as e:
            logger.warning("download.py 文件中处理失败，错误内容如下：")
            logger.warning(e)
            return 500