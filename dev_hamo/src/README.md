# AutopyClass
- 마우스, 키보드 제어 모듈

# GestureModule
- `trans_input` 함수
    - 학습된 모델의 input을 맞추기 위한 함수
    - RNN과 CNN의 input data의 형태가 다르므로 구분
    - 저장 이미지를 위해 input data와 학습용 데이터 반환
    - 모델 구분 선택 안하면 except 발생

- `predict` 함수
    - 학습된 모델로 예측하는 함수
    - 예측한 라벨과 예측률 반환