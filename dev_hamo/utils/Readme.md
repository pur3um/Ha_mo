# draw Module
## 함수 정리
- `draw_canvas`
    - 전달 받은 좌표로 이미지 데이터 생성

    ![image](https://user-images.githubusercontent.com/77317312/123464344-a5489c80-d627-11eb-80ab-05751f91d1d6.png)
- `save_file`
    - image 파일 혹은 csv 파일을 저장해주는 모듈
    - 모델이 예측한 예측률이 75% 이상이면 예측한 라벨의 디렉토리로 저장
    - 파일명이 중복되는 것을 방지하기 위해 현재 날짜와 시간(초)으로 파일명 저장

# hand_tracking_module
## Class 및 함수 정리
- media pipe 프레임워크를 기반 손 모션 인식 모듈
- ![image](https://user-images.githubusercontent.com/77317312/123464998-872f6c00-d628-11eb-9c3f-c506c7a84866.png)
- `find_hands`함수
    - image에서 손을 인식하고 각 마디별 landmarks 찍어주는 함수

- `find_positoin`함수
    - 손이 인식되면 마디 마다의 landmarks의 좌표를 저장, 반환해주는 함수
    - 손의 전체적인 bounding box를 그려주는 함수

- `fingers_up`함수
    - 각 손가락을 접었는지 폈는지를 판단해주는 함수
    - landmark의 특정 값의 변화를 이용하여 판단