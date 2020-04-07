#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: c:\deepsight-back\deepsight\management\commands\ai.py
# Project: c:\deepsight-back\deepsight
# Created Date: Sunday, December 22nd 2019, 2:38:54 pm
# Author: Zenturio
# -----
# Last Modified: Sun Dec 22 2019
# Modified By: Zenturio
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

from django.core.management.base import BaseCommand
from deepsight.main import DeepsightMain
import multiprocessing
from multiprocessing import Pool

def init_processpool():
    processlist = []
    main = DeepsightMain()
    mainprocess = multiprocessing.Process(main.run())
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