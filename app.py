from flask import Flask, request, jsonify
import time
import tensorflow as tf
import urllib.request
import tempfile
import os
import yaml
from io import BytesIO
#boto3는 python AWS SDK이다.
import boto3
from PIL import Image
from botocore.exceptions import NoCredentialsError
import numpy as np
from ultralytics import YOLO
import cv2

app=Flask(__name__)

with open('env.yml','r') as stream:
    env = yaml.safe_load(stream)

aws_access_key_id=env['aws_access_key_id']
aws_secret_access_key=env['aws_secret_access_key']
bucket_name=env['bucket_name']
region_name=env['region_name']

Mango_resultList=['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
Sugarcane_resultList=['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

@app.route("/mango",methods=['POST'])
def mango():
    try:
        #json 요청으로부터 s3url 얻기
        data=request.get_json()
        s3url=data.get('s3url')
        key=list(map(str,s3url.split("/")))[3]

        #세션 셜정
        session=boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        print("all")
        # s3 버킷에서 이미지 가져오기
        #
        # image=tf.keras.preprocessing.image.load_img(s3url, target_size=(224,224,3))
        # img_array=tf.keras.preprocessing.image.img_to_array(image)
        # img_array=np.expand_dims(img_array,axis=0)

        s3=session.resource('s3')
        object=s3.Object(bucket_name,key).get()
        file_stream=object['Body'].read()
        img=Image.open(BytesIO(file_stream))
        img = img.convert('RGB')

        #yolo
        model_Y = YOLO("best.pt")
        results = model_Y.predict(img)
        bbox = results[0].boxes[0].xyxy[0]
        x_min, y_min, x_max, y_max = bbox[:4]
        cropped_image = results[0].orig_img[0][int(y_min):int(y_max), int(x_min):int(x_max)]
        image_np = np.array(cropped_image)

        image_np = np.uint8(image_np.reshape((1, -1, 3)))
        #
        # # 이미지를 원하는 크기로 resize
        cropped_image_resized = cv2.resize(image_np, (224, 224))

        img_array = np.array(cropped_image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        model = tf.keras.models.load_model("resnet50_Test.h5")

        # 머신러닝 추론 수행
        predictions = model.predict(img_array)

        resultSet = np.array(predictions)

        # 가장 높은 확률을 가진 top3 질병 리스트화하여 리턴하기
        set = np.argsort(resultSet[0])[::-1]
        top3 = set[:3]
        result = []
        for idx in top3:
            result.append(Mango_resultList[idx])
        print(result)
        #
        # img=img.resize((224,224))
        # #
        # # img.save(byteImgIo,"PNG")
        # # byteImgIo.seek(0)
        # # img=byteImgIo.read()
        # print("all")
        # #
        # # 이미지를 모델에서 쓸 수 있도록 array로 변환하기
        # img_array=np.array(img)
        # img_array=np.expand_dims(img_array,axis=0)
        # print("all")
        #
        #
        # # 머신러닝 추론 수행
        # prediction=model.predict(img_array)
        # resultSet=np.array(prediction)
        # print("slkjf")
        #
        # # 가장 높은 확률을 가진 top3 질병 리스트화하여 리턴하기
        # set=np.argsort(resultSet[0])[::-1]
        # top3=set[:3]
        # result=[]
        # for idx in top3:
        #     result.append(Mango_resultList[idx])
        # print(result)

        return jsonify(result)
    except Exception as e:
        print(e)
        return "error"

# @app.route("/sugarcane", methods=['POST'])
# def sugarcane():
#     try:
#         # json 요청으로부터 s3url 얻기
#         # data = request.get_json()
#         # s3url = data.get('s3url')
#         # print("yest")
#         # print(data)
#         # print(s3url)
#         # key = list(map(str, s3url.split("/")))[3]
#         key="sugarcane_sample.jpeg"
#
#         #model 가져오기
#         model=tf.keras.models.load_model("C:/Users/yujin/Downloads/sugarcane_resnet50.h5")
#
#         # 세션 셜정
#         session = boto3.Session(
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=region_name
#         )
#
#         # s3 버킷에서 이미지 가져오기
#         s3 = session.resource('s3')
#         object = s3.Object(bucket_name, key).get()
#         file_stream = object['Body']
#         img = Image.open(file_stream)
#         img = img.resize((224, 224))
#
#         #이미지를 모델에서 쓸 수 있도록 array로 변환하기
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#
#         # 머신러닝 추론 수행
#         prediction = model.predict(img_array)
#         resultSet = np.array(prediction)
#
#         # 가장 높은 확률을 가진 top3 질병 리스트화하여 리턴하기
#         set = np.argsort(resultSet[0])[::-1]
#         top3 = set[:3]
#         result = []
#         for idx in top3:
#             result.append(Sugarcane_resultList[idx])
#         print(result)
#         return jsonify(result)
#     except Exception as e:
#         return "error"


if __name__=="__main__":
    app.run(port=8083, host="0.0.0.0")