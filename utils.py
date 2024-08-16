import os

import boto3
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from dotenv import load_dotenv

# 환경 변수 다운로드
load_dotenv()


def download_image(object_key):
    """S3에서 이미지를 다운로드하고 OpenCV 이미지로 변환 (최적화 버전)"""

    s3 = boto3.client(
        service_name="s3",
        region_name=os.getenv("S3_REGION_NAME"),  # 버킷 region
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),  # 액세스 키 ID
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")  # 비밀 액세스 키
    )

    try:
        # 스트리밍 응답 사용
        response = s3.get_object(Bucket=os.getenv("S3_BUCKET_NAME"), Key=object_key)
        stream = response['Body']

        # 스트림에서 직접 이미지 디코딩
        file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode image: {object_key}")
        print("Success to decode image from S3")
        return image
    except Exception as e:
        print(f"Error downloading image from S3: {e}")
        return None


def create_texture_info(image):
    # 이미지 로드
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 마스크 생성: 검정색 배경을 제외
    mask = cv2.inRange(image, (1, 1, 1), (255, 255, 255))

    # 관심 영역만 추출 (검정색 배경 제외)
    roi = gray[mask != 0]  # 1차원 배열인 상태

    # GLCM 계산을 위한 형태로 변환
    roi_reshaped = roi.reshape(-1, 1)

    # GLCM 계산
    distances = [1]
    angles = [np.pi / 2]
    glcm = graycomatrix(roi_reshaped, distances, angles, 256, symmetric=True, normed=True)

    # 텍스처 특징 추출
    texture_result = {
        "contrast": graycoprops(glcm, 'contrast')[0][0],
        "dissimilarity": graycoprops(glcm, 'dissimilarity')[0][0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0][0],
        "energy": graycoprops(glcm, 'energy')[0][0],
        "correlation": graycoprops(glcm, 'correlation')[0][0]
    }
    print("Success to extract texture info from image")
    return texture_result
