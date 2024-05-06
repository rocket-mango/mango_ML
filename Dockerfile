# 베이스 이미지 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
COPY app.py .
COPY resnet50_Test.h5 .
COPY best.pt .
# 필요한 경우 wget 설치
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# TensorFlow .whl 파일 다운로드 및 설치
RUN wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.15.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install tensorflow_cpu-2.15.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# 필요한 라이브러리 설치
RUN pip install -r requirements.txt

# Flask 애플리케이션 실행을 위한 명령어 설정
CMD ["python", "app.py"]
