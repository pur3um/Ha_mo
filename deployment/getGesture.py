import handTrackingModule as htm
import gestureModelModule as gmm
from tensorflow import keras
import cv2
from windowControl import WindowController

class Gesture():
    def __init__(self):
        # 웹캡 사이즈 설정 변수
        self.hCam, self.wCam = 640, 640     
        # 필요한 변수
        self.draw_arr = []
        self.in_check = 0
        self.out_check = 0
        self.pyauto = WindowController()
        # 모델 관련 변수
        self.model_selection = 'CNN'
        self.conf_limit = 0.75
        self.detector = htm.HandDetector(maxHands=1, detectionCon=0.75)
        self.gesture_model = keras.models.load_model('model/VGGColab-2021-07-06_08-26-05.h5')

    def predict(self, img, modeChange):
        # 손 인식시
        img = self.detector.findHands(img)
        self.landmark_list, _ = self.detector.findPosition(img, draw=False)
        action = ''
        imgCanvas = None
        control_mode = False
        if self.landmark_list:
            self.out_check = 0
            self.fingers = self.detector.fingersUp()
            # 검지가 펴졌을때만 control mode
            control_mode = (self.fingers[1]==1) and (1 not in self.fingers[2:])
            if control_mode:
                self.draw_arr.append(self.landmark_list[8][1:])
                cv2.circle(img, tuple(self.landmark_list[8][1:]), 7, (255,0,0), cv2.FILLED)
        else:
            self.out_check += 1
            if self.out_check == 10:
                if 30 < len(self.draw_arr) <= 100:
                    # 저장한 좌표로 input 데이터 생성
                    # 모델 추론
                    self.draw_arr = self.draw_arr[10:-7] # 앞, 뒤 10 frame 씩 제외 ## -5
                    input_data, imgCanvas = gmm.trans_input(self.draw_arr, self.wCam, self.hCam, self.model_selection)
                    pred, confidence = gmm.predict(self.gesture_model, input_data)
                    
                    if confidence > self.conf_limit:
                        modes = [self.pyauto.youtube, self.pyauto.webMode, self.pyauto.presentMode]
                        action = modes[modeChange](pred)
                self.draw_arr.clear()

        return control_mode, action, imgCanvas