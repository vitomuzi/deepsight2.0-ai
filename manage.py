#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: c:\deepsight-back\manage.py
# Project: c:\deepsight-back
# Created Date: Sunday, December 19nd 2019, 4:30:54 pm
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
import os
import sys
import logging
from django.apps import apps
from django.conf import settings
import environconfig


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "environconfig.settings.local")

    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        # The above import may fail for some other reason. Ensure that the
        # issue is really that Django is missing to avoid masking other
        # exceptions on Python 2.
        try:
            import django
        except ImportError:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            )
        raise
    execute_from_command_line(sys.argv)
