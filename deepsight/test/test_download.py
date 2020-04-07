#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: test_download.py
# Project: deepsight-back
# Created Date: 一月 07,2020 09:26
# Author: yanlin
# Description: 
# -----
# Last Modified: 2020/1/7 9:26
# Modified By: yanlin
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
'''

from deepsight.download import download
import json

tags = "tagging_xml"
code = "HN00001"

urls = '[{"imageUrl":"http: //xxxx/1.jpg","imageXML":"http: //xxxx/1.xml"},{"imageUrl":"http: //xxxx/10.jpg","imageXML":"http: //xxxx/10.xml"}]'
urls = json.loads(urls)


download.download_jpg_xml(urls, code)