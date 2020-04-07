#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: encoder.py
# Project: deepsight-back
# Created Date: 十二月 25,2019 12:49
# Author: vito
# Description: 编码器，和mq中的 producer配合使用
# -----
# Last Modified: 2019/12/25 12:49
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
from deepsight.network.mq import producer
from django.conf import settings
import logging

logger = logging.getLogger(settings.ENVTYPE)
# 提供接口供ai程序调取发布消息到rabbitmq
def encoder(body):
    producer_job = producer.Producer(body)
    logger.info("producer is running")
    producer_job.producer()


