# python manager.py ai启动consumer
# 该脚本测试后端back消费者能否正常消费
# python test-consumer.py默认发送测试数据到rabbitmq一次
# python test_consumer.py *  发送测试数据到rabbitmq*次


import pika
import time
import sys
import os

# 确保单独执行该脚本时，能够引入其他模块以及django的环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "environconfig.settings.local")
from django.conf import settings

body = '{"type":"tagging_request","data":{"code":"HN00001","imageUrls":["http://xxxx/1.jpg","http://xxxx/10.jpg","http://xxxx/111. jpg ","http://xxxx/130. jpg ","http://xxxx/190. jpg ","http://xxxx/30.jpg","http://xxxx/50. jpg ","http://xxxx/60. jpg ","http://xxxx/80. jpg","http://xxxx/90. jpg "]}}'
test_body = '{"data":{"code":"HN00001","imageUrls":["http://xxxx/1.jpg","http://xxxx/10.jpg","http://xxxx/111. jpg ","http://xxxx/130. jpg ","http://xxxx/190. jpg ","http://xxxx/30.jpg","http://xxxx/50. jpg ","http://xxxx/60. jpg ","http://xxxx/80. jpg","http://xxxx/90. jpg "]}}'

# body = "vito"
# body = json.loads(body1)
class Producer:
    def __init__(self, message):
        self.message = message
    def producer(self):
        try:
            credentials = pika.PlainCredentials(settings.RABBITMQ_USERNAME, settings.RABBITMQ_PASSWORD)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(settings.RABBITMQ_HOST, settings.RABBITMQ_PORT, settings.RABBITMQ_VHOST, credentials))
            channel = connection.channel()
            channel.exchange_declare(exchange=settings.RABBITMQ_EXCHANGE,
                                    exchange_type=settings.RABBITMQ_EXCHANGE_TYPE,
                                    durable=True)
            severity = 'deepsight.test'
            channel.basic_publish(exchange=settings.RABBITMQ_EXCHANGE,
                                routing_key=severity,
                                body=self.message,
                                properties=pika.BasicProperties(
                                delivery_mode=2,
                                ))
            print(" [x] Sent %r:%r" % (severity, self.message))
            connection.close()
        except Exception as result:
            print("发送消息失败，检查连接rabbitmq参数是否有误或者rabbitmq是否启动 \n {}".format(result))
            os._exit(1)

    def encoder(self):
        Producer.producer(self)

# 定义循环发送多少条测试数据
if len(sys.argv) == 1:
    Producer(body).encoder()
    print("已发送1次")
else:
    try:
        for i in range(int(sys.argv[1])):
            Producer(body).encoder()
        print("已发送{}次".format(int(sys.argv[1])))
    except ValueError:
        print("参数需要为正整数")



