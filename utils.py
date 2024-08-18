import io
import os

import boto3
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from dotenv import load_dotenv

# 환경 변수 다운로드
load_dotenv()


def ndarray_to_image(s3_conn, response, image_name):
    # Min-Max scaling을 통해 값을 [0, 255] 범위로 조정
    min_val = response.min()
    max_val = response.max()
    scaled_array = (response - min_val) / (max_val - min_val) * 255
    image = Image.fromarray(scaled_array.astype(np.uint8))

    # 이미지 데이터를 바이트 배열로 변환
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # AWS S3에 업로드
    bucket_name = s3_conn.bucket  # 버킷 이름

    # 업로드
    s3_conn.upload_fileobj(buffer, bucket_name, image_name)

    # 업로드된 이미지의 URL 생성
    image_url = f"https://{bucket_name}.s3.amazonaws.com/{image_name}"

    return image_url


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


def lbp_calculate(s3_conn, image, meat_id, seqno):
    # 이미지가 컬러 이미지인 경우 그레이스케일로 변환
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # LBP 결과를 저장할 배열 생성
    lbp1 = np.zeros_like(image)

    # 각 픽셀에 대해 LBP 계산
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # 중심 픽셀 값
            center = image[i, j]
            # 주변 8개 픽셀 값과 비교하여 이진 패턴 생성
            binary_pattern = 0
            binary_pattern |= (image[i - 1, j - 1] >= center) << 7
            binary_pattern |= (image[i - 1, j] >= center) << 6
            binary_pattern |= (image[i - 1, j + 1] >= center) << 5
            binary_pattern |= (image[i, j + 1] >= center) << 4
            binary_pattern |= (image[i + 1, j + 1] >= center) << 3
            binary_pattern |= (image[i + 1, j] >= center) << 2
            binary_pattern |= (image[i + 1, j - 1] >= center) << 1
            binary_pattern |= (image[i, j - 1] >= center) << 0
            # 결과 저장
            lbp1[i, j] = binary_pattern

    # Compute LBP
    radius = 3
    n_points = 8 * radius
    lbp2 = local_binary_pattern(image, n_points, radius, method='uniform')

    print("Success to create lbp images")

    # Save the LBP image
    image_name1 = f'openCV_images/{meat_id}-{seqno}-lbp1-{i + 1}.png'
    image_name2 = f'openCV_images/{meat_id}-{seqno}-lbp2-{i + 1}.png'

    print(image_name1, image_name2)

    lbp_image1 = ndarray_to_image(s3_conn, lbp1, image_name1)
    lbp_image2 = ndarray_to_image(s3_conn, lbp2, image_name2)

    result = {
        "lbp1": lbp_image1,
        "lbp2": lbp_image2
    }

    return result


def create_gabor_kernels(ksize, sigma, lambd, gamma, psi, num_orientations):
    kernels = []
    for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return kernels


def apply_gabor_kernels(img, kernels):
    responses = []
    for kernel in kernels:
        response = cv2.filter2D(img, cv2.CV_32F, kernel)
        responses.append(response)
    return responses


def compute_texture_features(responses):
    features = []
    for response in responses:
        mean = np.mean(response)
        std_dev = np.std(response)
        energy = np.sum(response ** 2)
        features.append([mean, std_dev, energy])
    return features


def gabor_texture_analysis(s3_conn, image, id, seqno):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gabor 필터 파라미터
    ksize = 31
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    num_orientations = 8  # 방향의 수

    kernels = create_gabor_kernels(ksize, sigma, lambd, gamma, psi, num_orientations)
    responses = apply_gabor_kernels(img, kernels)
    features = compute_texture_features(responses)
    print("Success to create gabor images")

    result = {}
    for i, response in enumerate(responses):
        image_name = f'openCV_images/{id}-{seqno}-garbor-{i + 1}.png'
        image_path = ndarray_to_image(s3_conn, response, image_name)

        result[i + 1] = {
            "images": image_path,
            "mean": float(features[i][0]),
            "std_dev": float(features[i][1]),
            "energy": float(features[i][2])
        }

    return result