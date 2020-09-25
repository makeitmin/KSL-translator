import cv2
import numpy as np
import time
import os

labelsPath = os.path.basename(os.getcwd()+'\\coco.names')
weightsPath = os.path.basename(os.getcwd()+'\\yolov3_final.weights')
configPath = os.path.basename(os.getcwd()+'\\yolov3_final.cfg')

LABELS = open(labelsPath).read().strip().split("\n")

# Load Yolo
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(LABELS), 3))

# Loading image
cap = cv2.VideoCapture(0)

# Preparing .mp4 file for the code
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
vout = cv2.VideoWriter()
success = vout.open('output.mp4', fourcc, 10, size, True)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

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
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            boxes = np.array(boxes)
            x, y, w, h = boxes[i]
            label = str(LABELS[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

            #(x,y) = (boxes[i][0], boxes[i][1])
            #(w,h) = (boxes[i][2], boxes[i][3])
            #crop_img = frame[y:y + h, x:x + h]

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

    # writing .mp4 file
    vout.write(frame)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)

    # Press Esc to finish
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()