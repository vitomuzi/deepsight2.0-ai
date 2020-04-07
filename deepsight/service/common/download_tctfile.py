#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: download_tctfile.py
# Project: deepsight-back
# Created Date: 十二月 27,2019 17:58
# Author: yanlin
# Description: 下载torch图
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
import requests
import urllib.request
from django.conf import settings
logger = logging.getLogger(settings.ENVTYPE)


def down_load_tct(url):

    logger.debug('--------------- '+url)
    path1 = settings.TCT_IMAGE
    #获取文件夹名
    tmp_url = url.split('/')[4]
    #组装成本地路径
    path2 = path1 + tmp_url +"/torch"
    #判断目录是否存在 不存在就创建
    if os.path.exists(path2):
        print('存在')
    else:
        os.makedirs(path2)
    path = path2+'/info.json'
    # url = 'http://192.168.1.47:12277/group1/JISNJGLSYN181101001/torch/info.json'
    #requests下载info.json文件
    response1 = requests.get(url)
    img = response1.content
    if response1.status_code == 200:
        logger.debug('download to '+url)
        with open(path, 'wb') as f:
            f.write(img)
    else:
        logger.debug('下载info.json失败')
        return 1001

    #获取url info.json以前的路径
    del_url = url[0:-9]
    logger.debug('start download analysis pictures')
    #下载info.json里面的图片
    with open(path, 'r') as f:
        for l in f:
            tup = l.rstrip('\n').rstrip().split('\t')
            arrret = tup[0]
            listArr = list(eval(arrret))
            for i in listArr:
                #组装下载路径
                sec_url = del_url+'%s' % i
                try:
                    response = urllib.request.urlopen(sec_url)
                    img = response.read()
                    with open(path2 + '/' + '%s' % i, 'wb') as code:
                        code.write(img)
                except Exception as e:
                    continue
                    logging.exception("Exception occurred")
    logger.debug('end download-----------')
    #返回状态码
    return 200
