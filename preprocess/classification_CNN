import cv2
from tensorflow.keras.models import load_model
import numpy as np
import string
import os

# 지화 라벨
labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14'] + ['del','nothing','space']
classes={}
i=0
for label in labels:
    classes[i] = label
    i+=1

# 지화 동작 CNN 모델 로드
model = load_model('분류 모델')
model.summary()

# 실시간 웹캠 할당
vid = cv2.VideoCapture(0)

# 실시간 분류 .mp4 녹화
size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
vout = cv2.VideoWriter('output_Classification.mp4', fourcc, 20, size, True)

while True:
    # 실시간 웹캠 프레임 읽어오기
    _, img = vid.read()
    ## new_im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 전체 화면에서 손 동작 영역 크롭
    image = img[50:400, 50:400]

    # 크롭된 이미지(image)의 height, width 할당 (color 은 필요 없어서 _ 처리)
    imageHeight, imageWidth, _ = image.shape

    # 이미지를 지화 CNN 모델에 넣어 예측
    image_skeleton = image_skeleton.reshape((1, 256, 256, 3))
    prediction = model.predict_classes(image_skeleton)[0]

    # 화면에 분류 결과 텍스트로 띄우기
    cv2.putText(img, classes[prediction], (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

    # 프로그램 실행하는 동안 .mp4 file 저장
    vout.write(img)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    # Esc 키로 종료
    if key == 27:
        break

vid.release()
vout.release()
cv2.destroyAllWindows()
