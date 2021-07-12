# YOLO

###### bdd_train.txt : 트레인 시킬 이미지들의 리스트
###### bdd_val.txt : validation 이미지 셋들의 리스트
###### kyjj.py : 모듈1. 영상에 간단한 사각형 출력한 모듈
###### kyjj2.py : 모듈2. blobfromimage로 yolo 작업한 모듈
###### kyjj3.py : 모듈3. dnndetectionmodel로 객체 검출한 모듈
###### kyjj4.py : 모듈4. kyjj3.py에 블러 처리한 모듈
###### kyjj5.py : 모듈5. gpu 설치후 만든 모듈
###### kyjj6.py : 모듈6. 모듈5에 크롭 추가
###### obj.data : class, train,valid,names,backup의 정보가 담긴 파일
###### obj.names : 검출할 객체들의 이름을 모아둔 파일
###### result1111.json : 테스트 시킨 결과를 저장한 json파일
###### yolo.ipynb : yolo 객체검출, cv2.dnn.readNet 사용, 파일리스트 출력, 다크넷 결과화면, 다크넷 명령어 정리
###### yolo_face.ipynb : yolo, cv2.dnn_DetectionModel 사용, 블러처리 추가, gpu사용법 확인
###### yolo_test_result.ipynb : yolo 학습시킨 weight파일로 test해서 나온 객체 결과 검출을 json파일로 정리



## 학습과정
1. 라벨링 : 각 이미지 당 바운딩 박스의 위치를 입력한 txt파일을 만들어서 라벨링을 한다.
2. 학습 : 라벨링 한 이미지를 바탕으로 학습을 진행한다. 학습을 하면 weights파일이 생성되는데 best_weight로 테스트를 진행하면 된다.
     - 다크넷 기본 세팅과 설정은 makefile에서 수정한다. 수정하면 'make'를 해줘야한다.
     - obj.data, obj.names 파일필요.
     - obj.data 파일에는 class, train,valid, names,backup의 정보가 담긴다. class에는 클래스 숫자. 나머지는 해당 파일의 위치정보이다.

3. 테스트 : 테스트를 하면 테스를 한 이미지 파일명, 포맷, 사이즈크기, 날짜, 객체 검출한 바운딩 박스 등 중요한 정보를 정리해서 json 파일로 만든다. 
4. 모듈 : 모델을 모듈 형태로 만든다. 예:) 인풋(일반 동영상) -> 아웃풋(객체검출, 블러처리 된 영상) 형태로 나오는 py파일 만들기
