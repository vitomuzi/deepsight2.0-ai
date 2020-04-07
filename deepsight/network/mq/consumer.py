'''
# File: consumer.py
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

from deepsight.network.coder import decoder
from django.conf import settings
import pika
import time
import logging

logger = logging.getLogger(settings.ENVTYPE)

"""
打包rabbitmq的consumer方法
"""

class Consumer:
    def __init__(self, conn):
        self.conn = conn

    def consumer(self):
        try:
            credentials = pika.PlainCredentials(settings.RABBITMQ_USERNAME, settings.RABBITMQ_PASSWORD)
            # 连接到rabbitmq服务器
            connection = pika.BlockingConnection(pika.ConnectionParameters(settings.RABBITMQ_HOST, settings.RABBITMQ_PORT, settings.RABBITMQ_VHOST, credentials, heartbeat=settings.RABBITMQ_HEARTBEAT))
            channel = connection.channel()
            # 提交交换机名称以及类型，durable声明队列持久化
            channel.exchange_declare(exchange=settings.RABBITMQ_EXCHANGE,
                                     exchange_type=settings.RABBITMQ_EXCHANGE_TYPE,
                                     durable=True)
            # 生成特定队列
            result = channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True)
            queue_name = result.method.queue
            severities = ['deepsight.#']
            # 将改特定队列与routing_key关键字以及exchange进行绑定
            for severity in severities:
                channel.queue_bind(exchange=settings.RABBITMQ_EXCHANGE,
                                    queue=queue_name,
                                    routing_key=severity)

            def callback(ch, method, properties, body):
                # logger.info(" [x] Received {}:{}".format(method.routing_key, body))
                decoder.decoder(body, self.conn)
                time.sleep(body.decode().count('.'))
                ch.basic_ack(delivery_tag=method.delivery_tag)

            # 确认一次从rabbitmq队列接收多少条消息
            channel.basic_qos(prefetch_count=5)

            # 持续接收消息
            channel.basic_consume(queue_name, callback, False)
            print(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        except ValueError:
            # 捕获传输数据规格错误的异常，并记录到日志
            logger.warning("从rabbitmq接收到的数据不符合规定类型！")
        except Exception as result:
            # 捕获pika操作rabbitmq的各种未知错误
            logger.warning("接收消息失败，检查连接rabbitmq参数是否有误或者rabbitmq是否启动 \n {}".format(result))
