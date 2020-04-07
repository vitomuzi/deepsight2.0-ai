#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: decoder.py
# Project: deepsight-back
# Created Date: 十二月 25,2019 12:50
# Author: vito
# Description: 解码器，和mq的 consumer 配合使用
# -----
# Last Modified: 2019/12/25 12:50
# Modified By: vito
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
'''
from django.conf import settings
import json
import logging
from deepsight.job.taskmanagement import TaskManagement

logger = logging.getLogger(settings.ENVTYPE)

"""
数据类型如下：
{
	"type": "tagging_request",
	"data":
	{
		"code": "HN00001",
		"imageUrls": [
			"http://xxxx/1.jpg",
			"http://xxxx/10.jpg",
			"http://xxxx/111.jpg ",
			"http://xxxx/130.jpg ",
			"http://xxxx/190.jpg ",
			"http://xxxx/30.jpg",
			"http://xxxx/50.jpg ",
			"http://xxxx/60.jpg ",
			"http://xxxx/80.jpg",
			"http://xxxx/90.jpg "
		]
	}
}
"""

def decoder(body, conn):
    try:
        # 解码从rabbitmq接收到的body
        bodys = json.loads(body)
        data_type = bodys["type"]
        code = bodys["data"]["code"]
        imageurls = bodys["data"]["imageUrls"]
    except:
        # 监测接收到的body是否符合规定，不符合抛出异常
        raise ValueError
    else:
        logger.info("type:{} \ncode:{} \nimageurls:{}".format(data_type, code, imageurls))
        # print(data_type, code, imageurls)
        # 调取ai接口跑ai程序
        task_management = TaskManagement(conn, code, data_type, imageurls)
        task_management.taskmanagement()





