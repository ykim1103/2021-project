import cv2
import sys


def video(file):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print('Video open failed')
        sys.exit()

    # 재생 파일 넓이와 높이
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  


    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('/data/kyj_dt/kyjj_output.mp4', fourcc, 30.0, (int(width), int(height)))



    while True:
        ret, frame = cap.read()
        if not ret :
            break    
        frame = cv2.rectangle(frame, (50, 200, 150, 100), (0, 255, 0), 2)     
        frame = cv2.rectangle(frame, (70, 220), (180, 280), (0, 128, 0), -1) 

        cv2.imshow('frame',frame)
        out.write(frame)   
        
    cap.release
    out.release()
    #out = cv2.VideoWriter('/data/kyj_dt/output.mp4', fourcc, 30.0, (int(width), int(height)))
    return out


# video("./mp4")
    #out = cv2.VideoWriter('/data/kyj_dt/output.mp4', fourcc, fps, (w, h))    