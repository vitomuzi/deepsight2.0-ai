#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: taskmanagement.py
# Project: deepsight-back
# Created Date: 十二月 25,2019 12:51
# Author: yanlin
# Description: 任务管理，任务调度
# -----
# Last Modified: 2020/01/08 14:23
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

from deepsight.service.tct.tct import tct
from deepsight.network.coder import encoder
from deepsight.service.tagging.tagging import Tagging
from deepsight.service.common.download_taggingxml import DownloadJpgXml


'''
class : TaskManagement 处理任务
number : 编号
tags :  type:tagging_request
urls : imageUrls：4x3 k的图片地址
'''
class TaskManagement:
    def __init__(self,conn, number, tags, urls):
        self.conn = conn
        self.number = number
        self.tags = tags
        self.urls = urls

    ''' 
        method : taskmanagement 不同的tags 处理不同的任务 
    '''
    def taskmanagement(self):
        if self.tags == 'tagging_request':
            #加载tagging 模块
            tag = Tagging(self.conn, self.number, self.urls)
            result_json = tag.tagging()
            # 调用编码器，处理分析好的数据
            encoder.encoder(result_json)

        elif self.tags == 'tagging_xml':
            #引入下载xml和jpg模块
            down_xml = DownloadJpgXml(self.conn, self.number, self.urls)
            down_xml.download_jpg_xml()

        else:
            for url in self.urls:
                tct.tct(url, self.number)