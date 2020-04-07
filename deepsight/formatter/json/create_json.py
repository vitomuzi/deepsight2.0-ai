#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: create_json.py
# Project: deepsight-back
# Created Date: 十二月 27,2019 15:28
# Author: yanlin
# Description: 生产json
# -----
# Last Modified: 2019/12/27 15:28
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
import json
import logging
from django.conf import settings
logger = logging.getLogger(settings.ENVTYPE)


'''
class : CreateTaggingJson  组装并转化json
slide:  编号
content: tagging 分析后的标记框
'''

class CreateTaggingJson:
    def __init__(self, slide, content):
        self.slide = slide
        self.content = content


    #tagging result exchange json
    def create_tagging_json(self):
        tagging_result = {
            "type": "tagging_result",
            "data": {
                "code": self.slide,
                "imageUrls": self.content
            }
        }
        #判断tagging 分析是否为空
        if self.content:
            # 转化为json串
            tagging_result = json.dumps(tagging_result)
        else:
            logger.warning('tagging analysis result is null')
        return tagging_result
