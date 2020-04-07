'''
# File: db.py
# Project: deepsight-back
# Created Date: 二月 11,2020 12:31
# Author: vito
# Description: 操作数据库增，改，查接口
# -----
# Last Modified: 2020/02/11 13:28
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

from django.core.management.base import BaseCommand
from deepsight.monitor import monitor_GPU
import multiprocessing
from multiprocessing import Pool


def init_processpool():
    processlist = []
    main = monitor_GPU.MonitorGPU()
    mainprocess = multiprocessing.Process(main.gpu_demo_show())
    processlist.append(mainprocess)
    mainprocess.start()
    # mainprocess.join()


class Command(BaseCommand):
    help = 'Start ai right now'

    # commands = [
    #     'python manage.py ai',

    #     'python manage.py runserver --'
    # ]

    def handle(self, *args, **options):
        init_processpool()