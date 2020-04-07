#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: c:\deepsight-back\deepsight\main.py
# Project: c:\deepsight-back\deepsight
# Created Date: Sunday, December 22nd 2019, 2:38:54 pm
# Author: Zenturio
# -----
# Last Modified: 2020/01/07 14:49
# Modified By: vito
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import os
import json
import logging
import pymysql
import time
from django.conf import settings
from threading import Thread
from deepsight.network.mq import consumer
from deepsight.network.mq import producer
from deepsight.monitor import monitor_GPU

logger = logging.getLogger(settings.ENVTYPE)
body = '挂起producer'

class DeepsightMain:
    # 连接数据库
    def db_connect(self):
        db = pymysql.connect(host=settings.DB_HOST, user=settings.DB_USERNAME,
                             password=settings.DB_PASSWORD, db=settings.DB_DATABASE, port=settings.DB_PORT,
                             charset='utf8mb4')
        return db

    # """
    # 循环检测连接数据库是否异常，如异常自动连接
    # """
    # def db_demo(self):
    #     while True:
    #         try:
    #             c.ping()
    #             logger.info("connect db successful")
    #         except:
    #             logger.warning("Database connection failed, retry connection!")
    #             c = DeepsightMain().db_connect()
    #         finally:
    #             time.sleep(30)

    def run(self):
        # 连接数据库
        c = DeepsightMain().db_connect()
        # Start ai here
        logger.info("start monitor_GPU")
        logger.info("start consumer")
        logger.info("start producer")
        # 调用monitor_GPU模块中的demo方法，持续监控GPU资源占比
        monitor_gpu_thread = monitor_GPU.MonitorGPU()

        # 调用consumer，producer方法，初始化两个线程
        consumer_thread = consumer.Consumer(c)
        producer_thread = producer.Producer(body)

        t1 = Thread(target=monitor_gpu_thread.GpuDemo, args=())
        t2 = Thread(target=consumer_thread.consumer, args=())
        t3 = Thread(target=producer_thread.demo, args=())
        # 循环检测连接数据库是否正常，异常重复连接
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
        c.close()

# class DeepsightMain:
#     def run(self):
#         # Start ai here
#         logger.info("start consumer")
#         # logger.info("start producer")
#         # 调用任务，模拟
#         consumer_thread = consumer.Consumer()
#         consumer_thread.consumer()

