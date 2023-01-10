from tensorflow.python.keras.preprocessing.image import img_to_array
from common import draw
import numpy as np
import cv2

def trans_input(draw_arr, x_size, y_size, model_selection):
        '''
        모델의 input값에 맞게 전처리
        input parameter 
            제스처 이동 좌표(draw_arr), 
            이미지 사이즈(width, height),
            모델 선택(CNN, RNN)
        return value
            ret == model input 값
            origin_canvas == 좌표대로 그린 그림
        '''
        try:
            # Canvas 초기화
            Canvas = np.zeros((y_size, x_size, 3), np.uint8)
            Canvas = draw.draw_canvas(Canvas, len(draw_arr), draw_arr)
            origin_canvas = Canvas.copy()
            model_selection = model_selection.upper()
            # CNN
            if model_selection == 'CNN':
                Canvas = cv2.resize(Canvas, (224, 224))
                Canvas = img_to_array(Canvas)
                Canvas = Canvas[np.newaxis, ...]
                ret = Canvas/255.
            # RNN
            elif model_selection == 'RNN':
                draw_arr += [[0, 0]] * (80 - len(draw_arr))
                input_arr = np.array(draw_arr)
                input_arr = input_arr / x_size
                ret = input_arr[np.newaxis, ...]
            return ret, origin_canvas
        except:
            # 모델을 선택하지 않는 경우 except
            print('모델을 선택하세요(CNN or RNN)')

def predict(gesture_model, input_data):
    '''
    학습된 모델로 예측
        예측한 값을 window_control module로 전달
        예측한 값과 예측률 반환
    '''
    # 예측
    pred = gesture_model.predict(input_data)

    # 예측 라벨과 예측률 추출
    idx = np.argmax(pred[0])
    maxRound = np.round(max(pred[0]), 2)

    print(maxRound, idx)
    return idx, maxRound