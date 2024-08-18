import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from consumer import create_consumer, async_process_image, process_messages
from producer import create_producer, send_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 환경 변수 다운로드
load_dotenv()

app = Flask(__name__)
CORS(app)


executor = ThreadPoolExecutor(max_workers=1)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/texture_info", methods=["POST"])
def create_meat_texture():
    producer = create_producer()

    data = request.json
    image_url = data.get('image_url')

    producer_msg = send_image(producer, image_url)

    time.sleep(2)

    # 비동기 작업 시작
    executor.submit(async_process_image, consumer)

    # producer 작업이 성공하면 즉시 200 반환
    return jsonify({"msg": f"{producer_msg['msg']} | Image processing started"}), 200


# Flask 실행
if __name__ == "__main__":
    consumer = create_consumer()
    # Kafka consumer를 별도의 스레드에서 실행
    kafka_thread = Thread(target=process_messages, args=(s3_conn, consumer))
    kafka_thread.daemon = True  # 메인 스레드가 종료되면 이 스레드도 종료됩니다
    kafka_thread.start()

    app.run(debug=True, host="0.0.0.0")
