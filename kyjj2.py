import cv2
import sys
import time
import numpy as np


def video(file):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print('Video open failed')
        sys.exit()

    # 재생 파일 넓이와 높이
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoo = cv2.VideoWriter('/data/kyj_dt/kyjj2_output.mp4', fourcc, fps, (int(width), int(height)))


    # yolo 적용
    net = cv2.dnn.readNet('yolov4.weights','yolov4.cfg')
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))



    while True:
        ret, img = cap.read()
    
        # 동영상 파일에서 캡쳐된 이미지를 이미지 파일 스트림으로 다시 인코딩을 한다. 
        #tmpStream = cv2.imencode(".jpeg", img)[1].tostring() 
        #wImg.value = tmpStream 
        

        # 20 프레임이 되기 위한 딜레이 다만, 실제로 입력한 것보다 조금 더 딜레이가 있다 
        time.sleep(0.05)
        ret, img = cap.read()

        if not ret :
            break

        #img = cv2.resize(img,None,fx=0.2,fy=0.2,interpolation = cv2.INTER_CUBIC)

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
    
        # 정보를 화면에 표시
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.65:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 좌표
                    x = int(center_x - w/2 )
                    y = int(center_y - h/2 )
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)    
        
        ## 노이즈제거
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)    


        #cv2.imshow('frame',img)
        
        videoo.write(img)   
        #cv2.imwrite(img)
    cap.release()
    videoo.release()
    return out


# video("./mp4")
    #out = cv2.VideoWriter('/data/kyj_dt/output.mp4', fourcc, fps, (w, h))    