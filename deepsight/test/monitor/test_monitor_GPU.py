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


import GPUtil
import time

# 获取gpu列表信息
Gpus = GPUtil.getGPUs()

def get_gpu_info():
    """
    :return:
    """
    gpulist = []
    GPUtil.showUtilization()

    # 获取多个GPU信息，存在列表
    for gpu in Gpus:
        print('GPU.id:', gpu.id)
        print('GPU总量:', gpu.memoryTotal)
        print('GPU使用量:', gpu.memoryUsed)
        print('GPU使用占比:', gpu.memoryUtil * 100)
        print('GPU.id:', gpu.id)
        # 按GPU逐个添加信息
        gpulist.append([gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])

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
    GPUavailable = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                       excludeID=[], excludeUUID=[])
    gpulist.append(GPUavailable)

    """
    根据GPU负载以及显存使用量返回第一个可用GPU_id，当无可用GPU时，将报错
    getAvailable参数均可用，含义一致
    attempts： 表示无法获取可用GPU时，尝试重复获取次数
    interval：  表示每次获取可用GPU时，时间间隔（秒）
    verbose:  表示在获取到最佳可用GPU时，是否打印尝试次数
    """
    GPUfirstavailable = GPUtil.getFirstAvailable(order='first', attempts=1, interval=900, verbose=False)

    gpulist.append(GPUfirstavailable)
    return gpulist

if __name__ == '__main__':
       gpu_info = get_gpu_info()
       print(gpu_info)
