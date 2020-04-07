import os
import json
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
TOTAL_PROCESSES = 2
ENVTYPE = ""
INSTALLED_APPS = [
    'deepsight'
]