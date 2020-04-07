'''
# File: db.py
# Project: deepsight-back
# Created Date: 二月 10,2020 14:31
# Author: vito
# Description: 操作数据库增，改，查接口
# -----
# Last Modified: 2020/02/10 15:00
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
import os
import logging
import GPUtil
import time
import psutil

logger = logging.getLogger(settings.ENVTYPE)

"""
通过gpu_utils获取GPU负载以及显存使用率最佳的gpu_id
from gpu_utils import gpu_init
gpu_id = gpu_init(best_gpu_metric="util")
mem_id = gpu_init(best_gpu_metric="mem")
print("gpu_id: {}   mem_id: {}".format(gpu_id, mem_id))
"""

stopped_num = 10000000  #(设置一个最大获取次数，防止记录文本爆炸)
delay = 0.5      #采集信息时间间隔

# 获取gpu列表信息
Gpus = GPUtil.getGPUs()

class MonitorGPU:
    def __init__(self, order='first', limit=1, max_load=0.5, max_memory=0.5, include_nan=False, exclude_id=[],
                 exclude_uuid=[], attempts=1, interval=900, verbose=False):
        self.order = order
        self.limit = limit
        self.max_load = max_load
        self.max_memory = max_memory
        self.include_nan = include_nan
        self.exclude_id = exclude_id
        self.exclude_uuid = exclude_uuid
        self.attempts = attempts
        self.interval = interval
        self.verbose = verbose

    def get_gpu_info_json(self):
            """
            :return:
            """
            result = {'GPU_info': []}

            # 获取多个GPU信息，存在列表
            for gpu in Gpus:
                GPU_info = dict()
                GPU_info['GPU.id'] = gpu.id
                GPU_info['GPU.uuid'] = gpu.uuid
                GPU_info['GPU总量'] = gpu.memoryTotal
                GPU_info['GPU使用量'] = gpu.memoryUsed
                GPU_info['GPU使用占比'] = "{:.1f}%".format(gpu.memoryUtil * 100)
                GPU_info['GPU负载'] = "{:.1f}%".format(gpu.load * 100)
                result['GPU_info'].append(GPU_info)
            return result

    def get_gpu_info_show(self):
        os.system('clear')
        GPUtil.showUtilization()

    def get_gpuavailable(self):

            """
            根据GPU负载以及显存使用量返回可用GPU_id列表
            first: 返回的gpu可用id按升序排列
            limit： 返回可用GPU的id数量
            maxload： GPU负载率最大限制（超过该值，将不会返回）
            maxMemory:  GPU显存使用率最大限制（超过该值，将不会返回）
            includeNan:  是否包括负载或内存使用为NaN的GPU
            excludeID:  排除的GPU_id列表
            excludeUUID：  类似excludeID，将ID替换成UUID
            """
            GPUavailable = GPUtil.getAvailable(order=self.order, limit=self.limit, maxLoad=self.max_load,
                                               maxMemory=self.max_memory, includeNan=self.include_nan,
                                               excludeID=self.exclude_id, excludeUUID=self.exclude_uuid)

            return GPUavailable

    def get_gpufirstavailable(self):

            """
            根据GPU负载以及显存使用量返回第一个可用GPU_id，当无可用GPU时，将报错
            getAvailable参数均可用，含义一致
            attempts： 表示无法获取可用GPU时，尝试重复获取次数
            interval：  表示每次获取可用GPU时，时间间隔（秒）
            verbose:  表示在获取到最佳可用GPU时，是否打印尝试次数
            """
            GPUfirstavailable = GPUtil.getFirstAvailable(order=self.order, attempts=self.attempts,
                                                         interval=self.interval, verbose=self.verbose)

            return GPUfirstavailable

    def GpuDemo(self):
            times = 0
            while True:
                    # 最大循环次数
                    if times < stopped_num:
                            # 获取GPU信息
                            gpu_info = MonitorGPU.get_gpu_info_json()
                            # 添加时间间隙
                            logger.info(gpu_info)
                            time.sleep(20)
                            times += 0.5
                    else:
                            break

    def gpu_demo_show(self):
            times = 0
            while True:
                    # 最大循环次数
                    if times < stopped_num:
                            # 获取GPU信息
                            MonitorGPU().get_gpu_info_show()
                            time.sleep(2)
                            times += 0.5
                    else:
                            break
