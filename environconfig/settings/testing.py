from .base import *

# enable debug mode
DEBUG = True
ALLOWED_HOSTS = ['localhost']
SECRET_KEY = "test(^v-lo$1fbse0*fg@p+@r31o7(&aoi3x%wuy=sy)ir6se-5p)$deep"
ENVTYPE = "testing"
BASE_LOG_DIR = "/var/log/ai"
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
             'filename': os.path.join(BASE_LOG_DIR, "all.log"),
             'backupCount': 3,  # 备份数为3  xx.log --> xx.log.1 --> xx.log.2 --> xx.log.3
             'formatter':'simple',
             'encoding': 'utf-8'
        },
        'logstash': {
            # 'level': 'DEBUG',
            # 'class': 'logstash.TCPLogstashHandler',
            # 'host': 'xxx.xxx.xxx.xxx',   # IP/name of our Logstash server
            # 'port': 5000,
            # 'version': 1,
            # 'message_type': 'logstash',
            # 'fqdn': True,
            # 'tags': ['ai'],
        }
    },
    'loggers': {
        'local': {
            'handlers': ['console' , 'file_handler', 'logstash'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}