#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: download_taggingfile.py
# Project: deepsight-back
# Created Date: 十二月 27,2019 17:58
# Author: yanlin
# Description: 下载4x3尺寸的原始图
# -----
# Last Modified: 2019/12/27 17:58
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
from deepsight.db import db
from deepsight.util import util
from django.conf import settings
from deepsight.download.download import Download
logger = logging.getLogger(settings.ENVTYPE)


'''
function : insert_picture_jpg 批量插入图片信息到数据库
conn : 数据库句柄
number : 该图片的编号
all_lists : 需要插入字段的列表
'''
def insert_picture_jpg(conn, number, all_lists):
    for all_list in all_lists:
        db.DbOperation(conn=conn, sample_id=number, url=all_list['url'], image_name=all_list['image_name']).db_insert()


'''
class : DownLoadTagging  下载图片
number 编号
urls  图片列表，下载图片的地址
'''
class DownLoadTagging:
    def __init__(self,conn, number, urls):
        self.conn = conn
        self.number = number
        self.urls = urls

    def down_load_tagging(self):
        # 组装分析下载地址
        save_folder = os.path.join(settings.TAGGING_IMAGE, self.number)
        #调用公共函数中的判断文件夹是否存在的函数
        util.check_folder(save_folder)
        # 创建一个空列表，收集这个样本所有的图片信息
        picture_lists = []

        logger.debug('start tagging picture download-----------')
        for url in self.urls:
            try:
                # 引入下载模块
                down_load = Download(save_folder, url)
                # result 是200 或者 500  picture_name 是 下载的文档名  save_picture_path 是保存在本地的地址
                result, picture_name = down_load.download()
                # 创建一个空字典，收集每一个图片的图片信息
                picture_dict = {}
                if result == 200:
                    picture_dict['url'] = url
                    picture_dict['image_name'] = picture_name
                    # 加入列表
                    picture_lists.append(picture_dict)
                else:
                    logger.warning("%s 没有下载成功" % url)
            except Exception as e:
                # 错误信息写入log
                logger.warning("download_taggingfile.py 文件中处理失败，错误内容如下：")
                logger.warning(e)
                continue
        logger.debug('end download-----------')
        #调用批量插入数据库
        insert_picture_jpg(self.conn, self.number, picture_lists)
        # 返回状态码
        return 200


