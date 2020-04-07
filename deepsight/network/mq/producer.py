'''
# File: producer.py
# Project: deepsight-back
# Created Date: 一月 7,2020 14:31
# Author: vito
# Description: 操作数据库增，改，查接口
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
'''

from django.conf import settings
import pika
import logging
import time
import os

logger = logging.getLogger(settings.ENVTYPE)

"""
打包rabbitmq的producer以及持久循环demo方法
"""

class Producer:
    def __init__(self, message):
        self.message = message
    def producer(self):
        try:
            credentials = pika.PlainCredentials(settings.RABBITMQ_USERNAME, settings.RABBITMQ_PASSWORD)
            # 连接到rabbitmq服务器
            connection = pika.BlockingConnection(pika.ConnectionParameters(settings.RABBITMQ_HOST, settings.RABBITMQ_PORT, settings.RABBITMQ_VHOST, credentials))
            channel = connection.channel()
            # 提交交换机名称以及类型，durable声明队列持久化
            channel.exchange_declare(exchange=settings.RABBITMQ_SEND_EXCHANGE,
                                    exchange_type=settings.RABBITMQ_EXCHANGE_TYPE,
                                    durable=True)
            severity = 'back.test'
            # 发布消息至该定义的交换机，且发布的消息携带的关键字routing_key前缀是back
            channel.basic_publish(exchange=settings.RABBITMQ_SEND_EXCHANGE,
                                routing_key=severity,
                                body=self.message,
                                properties=pika.BasicProperties(
                                    delivery_mode=2, # 使消息或任务也持久化存储
                                ))
            logger.info(" [x] Sent {}:{}".format(severity, self.message))
            # 关闭连接
            connection.close()
        except Exception as result:
            # 捕获pika操作rabbitmq的各种未知错误
            logger.warning("发送消息失败，检查连接rabbitmq参数是否有误或者rabbitmq是否启动 \n {}".format(result))
            # os._exit(1)

    # 持续循环demo方法，初始化producer线程
    def demo(self):
        while True:
            time.sleep(30)
            logger.info("*** producer is running ***")
            continue

