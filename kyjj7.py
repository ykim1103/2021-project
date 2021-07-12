import cv2
import sys
import time
import numpy as np
import os 
import datetime
import json
from collections import OrderedDict
import pandas as pd



def video(file):
    CONFIDENCE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


    # coconames 정의
    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    text_test = pd.read_csv('coco.names',header=None)
    text_test=list(np.array(text_test[0].tolist()))
    

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

    # 재생파일 이름과 확장자
    name = file.split('/')[-1]
    ffmo = name.split('.')[-1]

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoo = cv2.VideoWriter(f'/data/kyj_dt/kyjj7_{name}_output_.mp4', fourcc, fps, (int(width), int(height)))

    # 이미지생성시간
    create_time = os.path.getctime(file)
    create_timestamp = datetime.datetime.fromtimestamp(create_time)
    create_timestamp = create_timestamp.replace(microsecond=0)


    # 이미지 크기
    mysize = os.path.getsize(file)

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
  
        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        start_drawing = time.time()
        
        total_boxes = []
        for i in range(len(boxes)):
                # 객체검출
                frame2=frame[boxes[i][1]:boxes[i][1]+boxes[i][3],boxes[i][0]:boxes[i][0]+boxes[i][2]]
                cv2.imwrite(f'{name}_object_{i+1}.png',frame2)

                # boxes float에서 str로 전환
                list_int=list(map(float,boxes[i].tolist()))
                list_int=list(map(str,list_int))
                total_boxes.append(list_int) 


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

        # json만들기
        file_data = OrderedDict()
        
        file_data['FileInfo'] = {'Name':name,'Extension':ffmo,'Created':'UTC '+str(create_timestamp)+'.','FileSize':str(mysize),'FrameCount':str(fps)}
        file_data['AnnotationInfo']={'class':['person','car','truck'],'type':'bbox'}
        file_data['Annotation'] = {'frameNo':str(0),'width':str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ),'labels':[]}

        
        objs = file_data["Annotation"]["labels"]
        for i in range(len(boxes)):
            objs.append({'class':text_test[int(classes[i])],'type':'bbox','boxcorners':total_boxes[i],'id':str(i)})

        with open(f'kyjj7_{name}.json','w',encoding='utf-8') as make_file:
            json.dump(file_data, make_file,ensure_ascii=False,indent='\t')    


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