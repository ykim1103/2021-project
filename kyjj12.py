import cv2
import sys
import time
import numpy as np
import os 
import datetime
import json
from collections import OrderedDict
import pandas as pd
import random
import string
from Cryptodome.Cipher import AES
import glob
from Cryptodome.Util import Padding
import pandas as pd
import datetime
from Cryptodome import Random

## file
IMG_PATH_face = os.path.join(os.getcwd(),'kyjj_test_data')
IMG_PATH_face_list = glob.glob(IMG_PATH_face + '/*.mp4')

## dr
DR_PATH = os.path.join(os.getcwd(),'kyjj_test_data','output')

## yolo세팅
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(512,512), scale=1/255, swapRB=True,crop=False)



def key_generator(size=16, chars = string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))

key = key_generator(16)

class Cipher:
    def __init__(self, key, mode=AES.MODE_GCM):
        self.key = Padding.pad(str(key).encode(),16)
        self.mode = mode
        #print(self.key)
        
    def encrypt(self, crop_image):
        _, png_image = cv2.imencode(".png",crop_image)
        byte_image = png_image.tobytes()
        
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, self.mode, iv)
        
        return iv + cipher.encrypt(byte_image)

    
    def decrypt(self, enc_data):
        iv = enc_data[:16]
        cipher = AES.new(self.key, self.mode, iv)
        dec_result = cipher.decrypt(enc_data[16:])
        box_array = np.frombuffer(doc_result, dtype='uint8')
        box_image = cv2.imdecode(box_array, cv2.IMREAD_COLOR)
        
        return box_image
        
        



def video(file):
    #CONFIDENCE_THRESHOLD = 0.3
    #NMS_THRESHOLD = 0.4
    #COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


    # coconames 정의
    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    text_test = pd.read_csv('coco.names',header=None)
    text_test=list(np.array(text_test[0].tolist()))
    

    # yolo 적용
    # 네트워크 불러오기
    #net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

    # GPU 사용
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)


    # GPU 사용
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    #model = cv2.dnn_DetectionModel(net)
    #model.setInputParams(size=(512,512), scale=1/255, swapRB=True,crop=False)


    
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print('Video open failed')
        sys.exit()

    # 재생 파일 넓이와 높이
    width_2 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_2 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 재생파일 이름과 확장자
    name = file.split('/')[-1]
    ffmo = name.split('.')[-1]

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoo = cv2.VideoWriter(DR_PATH+f'/kyjj12_{name}_output_.mp4', fourcc, fps, (int(width_2), int(height_2)))

    # 이미지생성시간
    create_time = os.path.getctime(file)
    create_timestamp = datetime.datetime.fromtimestamp(create_time)
    create_timestamp = create_timestamp.replace(microsecond=0)


    # 이미지 크기
    mysize = os.path.getsize(file)
    
    # 리스트세팅
    frame_count = 0
    total_boxes22 = []
    total_class22 = []
    total_keys22 = []
    

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
  
        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        start_drawing = time.time()
        
        boxes_tem = []
        classes_tem = []
        key_tem = []
    

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
            
            # 암호화 처리
            aa = Cipher(bytes(key,encoding = 'utf-8')).encrypt(roi)
            
            # 프레임별 객체정보담기
            boxes_tem.append(box)
            classes_tem.append(classid)    
            key_tem.append(str(aa))
            
        total_boxes22.append(boxes_tem)
        total_class22.append(classes_tem)
        total_keys22.append(key_tem)
        frame_count +=1
        
        

        # meta json만들기
        meta_info = {'FileInfo':{'Name':name, 'Extension':ffmo, 'Created':'UTC '+str(create_timestamp)+'.', 'FileSize':str(mysize),'FrameCount':str(fps)},
                    'AnnotationInfo':{'class':['face'],'type':'bbox'},
                     'Annotation':{}}
        for i in range(len(total_boxes22)):
            meta_info['Annotation'][str(i)] = {'frameNo':str(i), 'width':str(width_2), 'height': str(height_2), 'labels':[]}
            objs = meta_info['Annotation'][str(i)]['labels']
            
            for j in range(len(total_boxes22[i])):
                objs.append({'class':text_test[int(total_class22[i][j])], 'type':'bbox','boxcorners':total_boxes22[i][j].tolist(), 'id':str(j)})
                
        # key json 만들기
        key_info = {}
        
        for i in range(len(total_keys22)):
            key_info[str(i)] = {}
            
            for j in range(len(total_keys22[i])):
                key_info[str(i)][j] = {'class':text_test[int(total_class22[i][j])], 'key':total_keys22[i][j],'boxcorners':total_boxes22[i][j].tolist()}
                
                
        with open(DR_PATH+f'/kyjj12_{name}_meta_info.json','w',encoding='utf-8') as make_file:
            json.dump(meta_info, make_file, ensure_ascii=False, indent='\t')

        with open(DR_PATH+f'/kyjj12_{name}_key_info.json','w',encoding='utf-8') as make_file:
            json.dump(key_info, make_file, ensure_ascii=False, indent='\t')    
            
        
        


        end_drawing = time.time()

    
        #cv2.imshow("detections", frame) 


        #cv2.imshow('frame',img)
        
        videoo.write(frame)   
        #cv2.imwrite(img)
    cap.release()
    videoo.release()
    return videoo


#file_path = []

for i in range(len(IMG_PATH_face_list)):
    start = time.time()
    
    #file_path.append(IMG_PATH_face_list[i])
    
    output = video(IMG_PATH_face_list[i])
    output_name = IMG_PATH_face_list[i].split('/')[-1]
    
    print(output_name)
    print("time : ", time.time()-start)
    print("--------------------------------------")

    
    
if __name__ =="__main__":
        try:
            app.run(video)
            
        except SystemExit:
            pass