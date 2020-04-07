from .base import *

# enable debug mode
DEBUG = True
ALLOWED_HOSTS = ['localhost']
SECRET_KEY = "test(^v-lo$1fbse0*fg@p+@r31o7(&aoi3x%wuy=sy)ir6se-5p)$deep"
ENVTYPE = "local"

RABBITMQ_HOST = "192.168.1.32"
RABBITMQ_PORT = "5672"
RABBITMQ_USERNAME = "admin"
RABBITMQ_PASSWORD = "Deepsight_0109"
RABBITMQ_VHOST = "/middle-back"
RABBITMQ_EXCHANGE = "deepsight_web"
RABBITMQ_QUEUE = "deepsight_web_topic"
RABBITMQ_EXCHANGE_TYPE = "topic"
RABBITMQ_HEARTBEAT = 60    #设置heartbeat心跳

RABBITMQ_SEND_QUEUE = "middle_back_topic"
RABBITMQ_SEND_EXCHANGE = "middle_back"

DB_HOST = "192.168.1.47"
DB_PORT = 12266
DB_USERNAME = "root"
DB_PASSWORD = "maxhealth1205"
DB_DATABASE = "backend"
DB_TABLE = "tag"

BASE_LOG_DIR = os.path.join(BASE_DIR, "log")

ZIP_FOLDER = "/opt/deepsight/ai/tag/zips"

UPLOAD_URL = "http://192.168.1.32:9000/images/tag"
# tagging 分析片子存放位置
TAGGING_IMAGE = "/opt/deepsight/ai/tag/images"

#存放tagging 医生标记好的的xml的地址
TAGGING_OUTPUT_IMAGE_XML = '/opt/deepsight/ai/tag/images'

#tct 分析片子存放位置
TCT_IMAGE = "/opt/deepsight/ai/tct/images"

#tct 细胞分析存放位置
TCT_CELL_IMAGE = "/opt/deepsight/ai/tct/cells"

#tct 病人分析存放位置
TCT_PATIENT_IMAGE = "/opt/deepsight/ai/tct/patient"



LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(asctime)s %(message)s'
        },
        'json': {
            'format': '{ "loggerName":"%(name)s", "timestamp":"%(asctime)s", "fileName":"%(filename)s", "logRecordCreationTime":"%(created)f", "functionName":"%(funcName)s", "levelNo":"%(levelno)s", "lineNo":"%(lineno)d", "time":"%(msecs)d", "levelName":"%(levelname)s", "message":"%(message)s"}',
        },
    },
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
        },
        'file_handler': {
             'level': 'DEBUG',
             'class': 'logging.handlers.TimedRotatingFileHandler',
             'filename': os.path.join(BASE_LOG_DIR, "local.log"),
             'backupCount': 3,  # 备份数为3  xx.log --> xx.log.1 --> xx.log.2 --> xx.log.3
             'formatter':'simple',
             'encoding': 'utf-8'
        },
    },
    'loggers': {
        'local': {
            'handlers': ['console' , 'file_handler'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
