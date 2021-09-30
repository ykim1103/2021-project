# YOLO

###### bdd_train.txt : 트레인 시킬 이미지들의 리스트
###### bdd_val.txt : validation 이미지 셋들의 리스트
###### data_check.ipynb : 메타데이터(동영상)정보 확인하는 코드
###### kyjj.py : 모듈1. 영상에 간단한 사각형 출력한 모듈
###### kyjj2.py : 모듈2. blobfromimage로 yolo 작업한 모듈
###### kyjj3.py : 모듈3. dnndetectionmodel로 객체 검출한 모듈
###### kyjj4.py : 모듈4. kyjj3.py에 블러 처리한 모듈
###### kyjj5.py : 모듈5. gpu 설치후 만든 모듈
###### kyjj6.py : 모듈6. 모듈5에 크롭 추가 (마지막 프레임만)
###### kyjj7.py : 모듈7. 모듈6에 json추가 (마지막 프레임만)
###### kyjj8.py : 모듈8. 모듈7에 매 프레임별로 캡쳐에서 저장하는 기능 추가, json부분 프레임카운트 수정(fps-> framecount), total_frame 이라는 변수에 모든 frame 저장 후 리턴, total_frame길이 출력
###### kyjj9.py : 모듀9. 모듈8에서 매프레임별 캡쳐해서 저장 삭제. 프레임별로 폴더 만들어서 그 안에 프레임별 객체 저장.
###### obj.data : class, train,valid,names,backup의 정보가 담긴 파일
###### obj.names : 검출할 객체들의 이름을 모아둔 파일
###### person_detection_file_copy.ipynb : xml에서 검출된것이 사람인것만 골라내는 코드, 파일 다른 곳으로 복사하는 코드
###### png to jpg.ipynb : png파일을 jpg로 변환
###### result1111.json : 테스트 시킨 결과를 저장한 json파일
###### train_valid_txt.ipynb : 학습시킬 사진리스트를 모아둔 txt파일 생성하는 코드
###### txt class_name_change.ipynb : txt에 있는 class를 1에서 0으로 변환
###### txt class_name_change.ipynb : txt에 class번호가 1로 되어 있는 것을 0으로 수정
###### txt to xml.ipynb : txt형식 라벨링파일을 xml로 변환
###### voc2yolo_converter.ipynb : yolo 학습을 위해 xml to txt 
###### yolo.ipynb : yolo 객체검출, cv2.dnn.readNet 사용, 파일리스트 출력, 다크넷 결과화면, 다크넷 명령어 정리
###### yolo_face.ipynb : yolo, cv2.dnn_DetectionModel 사용, 블러처리 추가, gpu사용법 확인
###### yolo_ipywidgets.ipynb : cv2.dnn_DetectionModel에 위젯 적용한 코드
###### yolo_test_result.ipynb : yolo 학습시킨 weight파일로 test해서 나온 객체 결과 검출을 json파일로 정리



## 학습과정
1. 라벨링 : 각 이미지 당 바운딩 박스의 위치를 입력한 txt파일을 만들어서 라벨링을 한다.
2. 학습 : 라벨링 한 이미지를 바탕으로 학습을 진행한다. 학습을 하면 weights파일이 생성되는데 best_weight로 테스트를 진행하면 된다.
     - 다크넷 기본 세팅과 설정은 makefile에서 수정한다. 수정하면 'make'를 해줘야한다.
     - obj.data, obj.names 파일필요.
     - obj.data 파일에는 class, train,valid, names,backup의 정보가 담긴다. class에는 클래스 숫자. 나머지는 해당 파일의 위치정보이다.

3. 테스트 : 테스트를 하면 테스를 한 이미지 파일명, 포맷, 사이즈크기, 날짜, 객체 검출한 바운딩 박스 등 중요한 정보를 정리해서 json 파일로 만든다. 
4. 모듈 : 모델을 모듈 형태로 만든다. 예:) 인풋(일반 동영상) -> 아웃풋(객체검출, 블러처리 된 영상) 형태로 나오는 py파일 만들기
5. 주의 사항 : - 다크넷 학습 시 이미지는 png 사용 불가 하다. 다크넷으로 학습할 때는 jpg이미지 형태를 사용한다.



## 기타 사항
1. voc2yolo_converter.ipynb 사용 시 'annotation'이라는 폴더를 생성하고 그 안에 'jpg'와 'xml'파일을 두면 'yolo'폴더에 txt파일이 생기고 'annotation.txt'에 주소리스트 생성.
- 참고 : https://bblib.net/entry/convert-voc-to-yolo-xml-to-yolo-xml-to-txt
