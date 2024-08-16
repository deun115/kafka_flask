import os
import pprint

from kafka import KafkaConsumer

from utils import *


load_dotenv()


# Create a KafkaConsumer instance
def create_consumer():
    return KafkaConsumer(
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=False
    )


def process_image(consumer):
    consumer.subscribe(topics=[os.getenv("KAFKA_TOPIC")])
    msg = consumer.poll(timeout_ms=1000)

    if msg:
        print("Success to Receive message")
        for message in msg.values():
            for m in message:
                pprint.pprint(m)
                s3_image = m.value.decode("utf-8")
                object_key = '/'.join(s3_image.split('/')[-2:])
                image = download_image(object_key)

                texture = create_texture_info(image)
                return texture

    return None


def async_process_image(consumer):
    texture = process_image(consumer)
    pprint.pprint(texture)
