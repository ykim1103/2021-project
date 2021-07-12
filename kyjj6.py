import cv2
import sys
import time
import numpy as np




def video(file):
    CONFIDENCE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    

    # yolo 적용
    # 네트워크 불러오기
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

    # GPU 사용
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    # GPU 사용
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(512,512), scale=1/255, swapRB=True,crop=False)


    
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print('Video open failed')
        sys.exit()

    # 재생 파일 넓이와 높이
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoo = cv2.VideoWriter('/data/kyj_dt/kyjj6_output.mp4', fourcc, fps, (int(width), int(height)))
    name = file.split('/')[-1]
    ffmo = name.split('.'[-1])



    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
  
        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        start_drawing = time.time()
        for i in range(len(boxes)):
                frame2=frame[boxes[i][1]:boxes[i][1]+boxes[i][3],boxes[i][0]:boxes[i][0]+boxes[i][2]]
                cv2.imwrite(f'{name}_object_{i+1}.png',frame2)


        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 블러처리
            left, top, width, height = box
            roi = frame[top:top+height, left:left+width]   # 관심영역 지정
            roi = cv2.GaussianBlur(roi,(15,15),0)
            frame[top:top+height, left:left+width] = roi   # 원본 이미지에 적용

        




        end_drawing = time.time()

        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        #cv2.imshow("detections", frame) 


        #cv2.imshow('frame',img)
        
        videoo.write(frame)   
        #cv2.imwrite(img)
    cap.release()
    videoo.release()
    return videoo