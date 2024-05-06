from flask import Flask, request
import time
import yaml
import tensorflow as tf
import urllib.request
import tempfile
import os
from io import BytesIO
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# boto3는 python AWS SDK이다.
import boto3
from PIL import Image
from botocore.exceptions import NoCredentialsError
import numpy as np
from ultralytics import YOLO
import cv2

with open('../env.yml','r') as stream:
    env = yaml.safe_load(stream)

aws_access_key_id=env['aws_access_key_id']
aws_secret_access_key=env['aws_secret_access_key']
bucket_name=env['bucket_name']
region_name=env['region_name']

Mango_resultList=['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
Sugarcane_resultList=['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

try:
    #json 요청으로부터 s3url 얻기
    # data=request.get_json()
    # s3url=data.get('s3url')
    # print("yest")
    # print(data)
    # print(s3url)
    key="79d5c660-d916-4fa4-a7ae-f00b2ea51b3e.png"

    #model 가져오기
    #세션 셜정
    session=boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    # s3 버킷에서 이미지 가져오기
    s3=session.resource('s3')
    object=s3.Object(bucket_name,key).get()
    file_stream=object['Body'].read()
    img=Image.open(BytesIO(file_stream))
    img = img.convert('RGB')

    model_Y=YOLO("../best.pt")
    print("here")
    results=model_Y.predict(img)

    bbox = results[0].boxes[0].xyxy[0]
    x_min, y_min, x_max, y_max = bbox[:4]  # 좌표를 각각의 변수에 할당
    # print("here")
    #
    # # # 원본 이미지 경로
    # image_path = results[0].path
    # print(results[0])
    # arr = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    # print("here")
    # print(file_stream)
    # image = cv2.imdecode(arr,-1)
    # print("here")

    # YOLO로 예측된 부분 잘라내기
    cropped_image = results[0].orig_img[0][int(y_min):int(y_max), int(x_min):int(x_max)]


    # print("here")
    #
    # ### resnet 돌리기
    # model = tf.keras.models.load_model("../resnet50_Test.h5")
    # print("here")
    #
    # # 이미지를 NumPy 배열로 변환
    image_np = np.array(cropped_image)

    image_np=np.uint8(image_np.reshape((1,-1,3)))
    #
    # # 이미지를 원하는 크기로 resize
    cropped_image_resized = cv2.resize(image_np, (224, 224))
    #
    # # 이미지를 정규화합니다.
    # cropped_image_normalized = cropped_image_resized / 255.0
    # print("here")
    #
    # # 모델에 입력 이미지를 전달하여 예측을 수행합니다.
    # predictions = model.predict(np.expand_dims(cropped_image_normalized, axis=0))
    # print(predictions)
    # print(cropped_image)
    #
    # cropped_image=cropped_image.resize((224,224))


    # 이미지를 모델에서 쓸 수 있도록 array로 변환하기
    img_array=np.array(cropped_image_resized)
    img_array=np.expand_dims(img_array,axis=0)

    print("slkj")

    model = tf.keras.models.load_model("../resnet50_Test.h5")

    # 머신러닝 추론 수행
    predictions=model.predict(img_array)

    resultSet=np.array(predictions)

    # 가장 높은 확률을 가진 top3 질병 리스트화하여 리턴하기
    set=np.argsort(resultSet[0])[::-1]
    top3=set[:3]
    result=[]
    for idx in top3:
        result.append(Mango_resultList[idx])
    print(result)
    print("ok")

except Exception as e:
    print(e)


