import os
import sys
import pymysql

# 确保单独执行该脚本时，能够引入其他模块以及django的环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "environconfig.settings.local")
from django.conf import settings
from deepsight.db import db

db_connect = pymysql.connect(host=settings.DB_HOST, user=settings.DB_USERNAME,
                     password=settings.DB_PASSWORD, db=settings.DB_DATABASE,
                     port=settings.DB_PORT, charset='utf8mb4')

db.DbOperation(conn=db_connect, sample_id="HN0004", start_time="2020-01-08 09:18:04",
               finish_time="2020-01-08 10:24:43", duration=457, is_completed=1, result="[13,243],[53,65]",
               image_name="wulei").db_update_results()
