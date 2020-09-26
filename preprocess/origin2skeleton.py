import cv2
import os

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
HAND_PARTS = {"Palm": 0,
              "TFirst": 1, "TSecond": 2, "TThird": 3, "TPoint": 4,
              "IFirst": 5, "ISecond": 6, "IThird": 7, "IPoint": 8,
              "MFirst": 9, "MSecond": 10, "MThird": 11, "MPoint": 12,
              "RFirst": 13, "RSecond": 14, "RThird": 15, "RPoint": 16,
              "PFirst": 17, "PSecond": 18, "PThird": 19, "PPoint": 20
              }

POSE_PAIRS = [["Palm","TFirst"], ["TFirst","TSecond"], ["TSecond","TThird"], ["TThird","TPoint"], # 엄지
              ["Palm","IFirst"], ["IFirst","ISecond"], ["ISecond","IThird"], ["IThird","IPoint"], # 검지
              ["Palm","MFirst"], ["MFirst","MSecond"], ["MSecond","MThird"], ["MThird","MPoint"], # 중지
              ["Palm","RFirst"], ["RFirst","RSecond"], ["RSecond","RThird"], ["RThird","RPoint"], # 약지
              ["Palm","PFirst"], ["PFirst","PSecond"], ["PSecond","PThird"], ["PThird","PPoint"] # 새끼
            ]

# 각 파일 path
protoFile = "OpenPose 프로토파일"
weightsFile = "OpenPose 가중치"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

for folder in range(1, 32):
    # 이미지 읽어오기 : 폴더 내의 파일 수를 n = len() 으로 받아와서 반복하여 이미지 호출
    n = len((os.walk('C:\\data\\'+str(folder)).__next__()[2]))
    for num in range(1,n+1):
        image = cv2.imread("C:\\data\\{}\\origin ({}).png".format(folder,num))

        # frame.shape = 불러온 이미지에서 height, width, color 받아옴
        imageHeight, imageWidth, _ = image.shape

        # network에 넣기위해 전처리
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

        # network에 넣어주기
        net.setInput(inpBlob)

        # 결과 받아오기
        output = net.forward()

        # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
        H = output.shape[2]
        W = output.shape[3]
        print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ", output.shape[3])  # 이미지 ID

        # 키포인트 검출시 이미지에 그려줌
        points = []
        for i in range(0, 21):
            # 해당 신체부위 신뢰도 얻음.
            probMap = output[0, i, :, :]

            # global 최대값 찾기
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # 원래 이미지에 맞게 점 위치 변경
            x = (imageWidth * point[0]) / W
            y = (imageHeight * point[1]) / H

            # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
            if prob > 0.1:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
                cv2.putText(image, "{}".format(""), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                            lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # cv2.imshow("Dots", image)
        # cv2.waitKey(0)

        # 이미지 복사
        skeletonImg = cv2.imread("black.png")

        # 각 POSE_PAIRS 별로 관절 드로잉 (손바닥-엄지첫마디, 엄지첫마디-엄지둘째마디, etc)
        for pair in POSE_PAIRS:
            partA = pair[0]  # Palm
            partA = HAND_PARTS[partA]  # 0
            partB = pair[1]  # TFirst
            partB = HAND_PARTS[partB]  # 1

            # print(partA," 와 ", partB, " 연결\n")
            if points[partA] and points[partB]:
                cv2.line(skeletonImg, points[partA], points[partB], (0, 255, 0), 2)

        cv2.imwrite("C:\\data_skeleton\\{}\\{}.png".format(folder,num), skeletonImg)
        # cv2.imshow("Skeleton", skeletonImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
