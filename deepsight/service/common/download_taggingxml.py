#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: download.py
# Project: deepsight-back
# Created Date: 一月 07,2020 09:09
# Author: yanlin
# Description: 下载标记过的jpg图和xml，并把状态写入数据库
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
import shutil
import logging
import urllib.request
from deepsight.db import db
from deepsight.util import util
from django.conf import settings
from deepsight.download.download import Download

logger = logging.getLogger(settings.ENVTYPE)

'''
function : get_db_urls  获取数据库中的某编号下的所有url地址
conn : 数据库连接句柄
number : 编号
'''
def get_db_urls(conn, number):
    db_results = db.DbOperation(conn=conn, sample_id=number).db_select()
    return db_results


'''
function : validate_database 验证某编号下的url字段是否存在 return True or None
url : url地址  用于判断是否存在数据库
db_results : 数据库中的值
'''
def validate_database(url, db_results):
    #循环遍历数据库
    for db_result in db_results:
        if db_result[1] == url:
            return True

'''
function : update_picture_xml 批量更新图片信息到数据库
conn : 数据库句柄
number : 该图片的编号
all_lists : 需要更新字段的列表
'''
def update_picture_xml(conn, number, all_lists):
    for all_list in all_lists:
        db.DbOperation(conn=conn, sample_id=number, url=all_list['imageUrl'], xml_downloaded=1, xml_name=all_list['picture_name'], xml_url=all_list['imageXML']).db_update_xml()


'''
class : DownloadJpgXml  下载系统标记比对过的jpg和对应的xml
urls : 列表  [{"imageUrl":"http: //xxxx/1.jpg","imageXML":"http: //xxxx/1.xml"}] 里面是图片和xml
number : 编号
'''
class DownloadJpgXml:
    def __init__(self,conn, number, urls):
        self.conn = conn
        self.number = number
        self.urls = urls


    def download_jpg_xml(self):

        # 组装分析下载地址
        save_folder = os.path.join(settings.TAGGING_OUTPUT_IMAGE_XML, self.number)

        #调用查询数据库
        db_results = get_db_urls(self.conn,self.number)
        #创建一个空列表，收集这个样本所有的图片信息
        picture_lists = []

        logger.debug('start tagging picture download-----------')
        for url in self.urls:
            try:
                imageUrl = url['imageUrl']
                imageXML = url['imageXML']
                results = validate_database(imageUrl, db_results)
                if results:
                    # 引入下载模块
                    down_load = Download(save_folder, imageXML)
                    # result 是200 或者 500  picture_name 是 下载的文档名  save_picture_path 是保存在本地的地址
                    result, picture_name = down_load.download()
                    # 创建一个空字典，收集每一个图片的图片信息
                    picture_dict = {}
                    # 更新数据库
                    if result == 200:
                        picture_dict['imageUrl'] = imageUrl
                        picture_dict['picture_name'] = picture_name
                        picture_dict['imageXML'] = imageXML
                        #加入列表
                        picture_lists.append(picture_dict)
                    else:
                        logger.warning("%s 没有下载成功" % imageXML)
                else:
                    logger.warning('No find '+imageUrl)
            except Exception as e:
                logger.warning("download_taggingxml.py 文件中处理失败，错误内容如下：")
                logger.warning(e)
                continue;

        #调用更新数据库
        update_picture_xml(self.conn, self.number, picture_lists)
        #下载好打包
        logger.debug('start zip  ------------')
        zip_folder = os.path.join(settings.ZIP_FOLDER, self.number)
        shutil.make_archive(zip_folder, 'zip', root_dir=save_folder)
        logger.debug('end zip-----------')