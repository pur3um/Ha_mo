# Model
- VGG16
- MobileNetV2
- LSTM

## process
- 각 모델 구축 과정 ipy note file
- colab에서 진행

## saved_model
- 학습 후 저장된 모델 파일

## preprocessing.py
- 학습용 데이터 생성 파일


## data directory
- 학습을 위한 데이터 directory
- 이전 비정제 데이터와 정제 데이터 보관
- temp == 앞으로 새로 생기는 데이터 저장 directory
- - -
# Data

## 구축목적
- 손동작을 인식하여 이를 이미지화하는 초기 모델의 데이터 생성하였다.
- 동작 인식에 활용되는 손가락의 개수를 다양화하여 동작 인식에 편의성 제공하기 위하여 구축하였다.

## 구축내용 및 데이터량
- 각 프레임에 인식된 손가락 좌표들을 담은 csv 파일, 좌표를 선으로 연결한 image 파일 생성하였다.
- 이미지 데이터 구축 수량: 300장 X 4
- CSV 데이터 구축 수량 :  300개 X 4(label)
- csv file 1개 == 제스처 데이터 1개
- 데이터 구조(size, 포맷, 도면)
- 이미지 파일명: 비정제데이터 + ‘/’ + 동작명 + ‘/’ + 생성일자 시간 + ‘.png’
- 데이터 Size
- Image: 640X360
- CSV: 20 rows X 4 columns

## 데이터 포맷
- image: png
- RNN 데이터: csv

## 디렉토리 경로
- img, csv 디렉토리 각각 해당
  
```bash
├── data/
│   ├── csv/           
│   │   ├── 비정제데이터 
│   │   │   ├── previous         - 각 동작별 디렉토리
│   │   │   ├── next
│   │   │   ├── double_previous
│   │   │   ├── double_next
│   │   │   └── temp
│   │   ├── 정제데이터
│   │   │   ├── 01next_img       - 각 동작별 디렉토리
│   │   │   ├── 02previous_img
│   │   │   ├── 03S_img
│   │   │   ├── 04W_img
│   │   │   └── temp
│   │   └── temp
│   └── img/
│       ├── 비정제데이터          - csv 디렉토리와 동일
│       ├── 정제데이터
│       └── temp
│
├── train/                       - 모델 학습
│   ├── backward
│   ├── fast_forward
│   └── forward
└── test/                        - 모델 검증 및 평가
    ├── backward
    ├── fast_forward
    └── forward 
```
