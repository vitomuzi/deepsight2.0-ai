# -*- coding: utf-8 -*-
import logging
import os
import time
#ai调用deepsight后端 修改分析结果
DEEPSITHT_IP_API = "http://192.168.1.32:13111/api/slide/updateSlide"
#DEEPSITHT_IP_API = "http://192.168.1.15:15040/api/slide/updateSlide"
#ccbackend-april2019中参数的绝对路径
ABSOLUTE_PATH = '/opt/test/tct/'

#分析好之后，删除文件的文件路径

DELETE_PATH = '/opt/image/'

#log path
LOG_PATH = '/opt/log'

#上传缩略图到case
HOST="http://192.168.1.32:13000/upload"
#HOST="http://192.168.1.47:12277/upload"

#后端ip
BACKEND_IP="http://192.168.1.32:13111/"
#BACKEND_IP="http://192.168.1.15:15040/"

# logging.basicConfig(
#     format="%(asctime)s = %(name)s - %(levelname)s - %(message)s",
#     filename=os.path.join(LOG_PATH,'ai-{}.log'.format(time.strftime('%Y-%m-%d'))),
#     level=logging.DEBUG,
#     filemode='a+'
# )
