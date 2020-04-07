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

db.DbOperation(conn=db_connect, sample_id="HN0004", url="http://192.168.1.32:9000/images/tag/04.jpg", xml_downloaded=1,
               xml_name="/opt/images", xml_url="http://192.168.1.32:9000/images/tag/03.xml").db_update_xml()
