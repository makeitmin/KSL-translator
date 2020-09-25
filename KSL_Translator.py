'''
Version : 2020/09/25

--사용하는 모델
YOLO : 손 탐지
OpenPose : 손 관절 그리기
CNN : 지화 동작 분류

--코드 플로우
1. 웹캠 실시간 영상을 받아 OpenCV를 통해 초당 프레임(frame)을 받아옵니다.
2. 초당 프레임에서 YOLO 모델을 활용하여 손을 탐지합니다.
3. 탐지한 손만 크롭되어 img 폴더에 저장됩니다.(crop_img)
4. img 폴더에 저장된 손 이미지들을 불러와서 OpenPose 모델을 활용하여 관절을 그립니다.
5. 관절을 그린 후 원본 이미지를 black.png 로 대체하여 배경을 검은색으로 교체한 뒤(image_skeleton) img_skeleton 폴더에 저장합니다.
6. img_skeleton 폴더에 저장된 관절 이미지들을 불러와서 CNN 모델을 활용하여 지화 동작을 분류합니다.

'''

import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model

# OpenPose MPII에서 손 관절의 각 파트 번호, 선으로 연결될 POSE_PAIRS
HAND_PARTS = {"Wrist": 0,
              "TFirst": 1, "TSecond": 2, "TThird": 3, "TPoint": 4,
              "IFirst": 5, "ISecond": 6, "IThird": 7, "IPoint": 8,
              "MFirst": 9, "MSecond": 10, "MThird": 11, "MPoint": 12,
              "RFirst": 13, "RSecond": 14, "RThird": 15, "RPoint": 16,
              "PFirst": 17, "PSecond": 18, "PThird": 19, "PPoint": 20
              }
POSE_PAIRS = [["Wrist","TFirst"], ["TFirst","TSecond"], ["TSecond","TThird"], ["TThird","TPoint"], # 엄지
              ["Wrist","IFirst"], ["IFirst","ISecond"], ["ISecond","IThird"], ["IThird","IPoint"], # 검지
              ["Wrist","MFirst"], ["MFirst","MSecond"], ["MSecond","MThird"], ["MThird","MPoint"], # 중지
              ["Wrist","RFirst"], ["RFirst","RSecond"], ["RSecond","RThird"], ["RThird","RPoint"], # 약지
              ["Wrist","PFirst"], ["PFirst","PSecond"], ["PSecond","PThird"], ["PThird","PPoint"] # 새끼
            ]

# YOLO 설정 파일 Path
labelsPath = os.getcwd()+"\\coco.names" # Hand 라벨
weightsPath = os.getcwd()+"\\yolov3_final.weights" # 가중치
configPath = os.getcwd()+"\\yolov3_final.cfg" # 모델 구성

# YOLO 라벨(hand) 호출
YOLO_LABELS = open(labelsPath).read().strip().split("\n")

# YOLO 모델 호출
yolo_net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# YOLO 출력층 설정
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(YOLO_LABELS), 3))

# OpenPose 설정 파일 Path
protoFile = os.getcwd()+"\\pose_deploy.prototxt" # 모델 구성
weightsFile = os.getcwd()+"\\pose_iter_102000.caffemodel" # 가중치

# OpenPose 모델 호출
caffe_net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# CNN 라벨 호출
cnn_labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14'] + ['del','nothing','space']
cnn_classes={}
i=0
for label in cnn_labels:
    cnn_classes[i] = label
    i+=1

# CNN 모델 호출
model = load_model('14_classifier_skeleton_20200906.h5')
model.summary()

# 실시간 웹캠 할당
cap = cv2.VideoCapture(0)

'''
.mp4 파일 생성 -> 코드 실행 속도 문제로 일단 주석 처리
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
vout = cv2.VideoWriter()
success = vout.open('output.mp4', fourcc, 10, size, True)
'''

# 화면 폰트
font = cv2.FONT_HERSHEY_PLAIN

# FPS 측정 변수
starting_time = time.time()
frame_id = 0
count=0

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                ww = int(detection[2] * width)
                hh = int(detection[3] * height)

                # Rectangle coordinates
                xx = int(center_x - ww / 2)
                yy = int(center_y - hh / 2)

                boxes.append([xx, yy, ww, hh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                crop_img = frame[yy:yy + hh, xx:xx + hh]
                cv2.imwrite(os.getcwd()+"\\img\\hand_"+str(count)+".jpg", crop_img)

                # 크롭된 이미지(crop_img)의 height, width 할당 (color 은 필요 없어서 _ 처리)
                imageHeight, imageWidth, _ = crop_img.shape

                # crop_img 를 OpenPose dnn network에 넣기위해 전처리
                inpBlob = cv2.dnn.blobFromImage(crop_img, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

                # crop_img를 OpenPose network에 넣어주기
                caffe_net.setInput(inpBlob)

                # OpenPose에 넣은 결과 받아오기
                opp_output = caffe_net.forward()

                # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
                H = opp_output.shape[2]
                W = opp_output.shape[3]
                print("이미지 ID : ", len(opp_output[0]), ", H : ", opp_output.shape[2], ", W : ", opp_output.shape[3])  # 이미지 ID

                # 키포인트 검출시 이미지에 드로잉
                points = []
                for i in range(0, 21):
                    # 해당 신체부위 신뢰도
                    probMap = opp_output[0, i, :, :]

                    # global 최대값 찾기
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                    # 원래 이미지에 맞게 점 위치 변경
                    x = (imageWidth * point[0]) / W
                    y = (imageHeight * point[1]) / H

                    # 키포인트 검출한 결과가 0.1보다 크면 (검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
                    if prob > 0.1:
                        cv2.circle(crop_img, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                                   lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
                        cv2.putText(crop_img, "{}".format(""), (int(x), int(y)), font, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                        points.append((int(x), int(y)))
                    else:
                        points.append(None)

                # 검은색 배경 이미지에 관절 드로잉하기 위해 black.png 호출
                img_skeleton = cv2.imread("black.png")

                # 각 POSE_PAIRS 별로 관절 드로잉 (손바닥-엄지첫마디, 엄지첫마디-엄지둘째마디, etc)
                for pair in POSE_PAIRS:
                    partA = pair[0]  # Palm
                    partA = HAND_PARTS[partA]  # 0
                    partB = pair[1]  # TFirst
                    partB = HAND_PARTS[partB]  # 1

                    # 연결 로그 출력 - print(partA," 와 ", partB, " 연결\n")
                    if points[partA] and points[partB]:
                        cv2.line(img_skeleton, points[partA], points[partB], (0, 255, 0), 2)

                # 실시간 관절 출력
                cv2.imshow('Img_Skeleton', img_skeleton)

                # 폴더 저장
                cv2.imwrite(os.getcwd()+"\\img_skeleton\\hand_"+str(count)+".jpg", img_skeleton)
                count += 1

                # 검은색 배경 이미지에 드로잉된 관절을 지화 CNN 모델에 넣어 예측
                img_skeleton = img_skeleton.reshape((1, 256, 256, 3))
                prediction = model.predict_classes(img_skeleton)[0]

                # 화면에 분류 결과 텍스트로 띄우기
                cv2.putText(frame, cnn_classes[prediction], (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    
    # 화면에 손 탐지 바운딩 박스 그리기
    for i in range(len(boxes)):
        if i in indexes:
            boxes = np.array(boxes)
            x, y, w, h = boxes[i]
            label = str(YOLO_LABELS[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

    # FPS 측정
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

    # .mp4 파일 저장
    # vout.write(frame)

    # 디스플레이
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)

    # ESC 키로 종료
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()