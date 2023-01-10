import sys
from common import HandTrackingModule as htm
from src import GestureModelModule as gmm
from src import AutopyClass
from tensorflow import keras
from common import draw
import numpy as np
import cv2
from src.user_interface import webcam_ij as wb
# from src.user_interface.webcam import DryHand as dr
    
class demopy():
    def __init__(self):
        ###########################################
        # 모델 추론시 변수 초기 셋팅
        self.draw_arr = []
        self.in_check = 0
        self.out_check = 0
        self.control_mode = False
        self.model_selection = 'CNN'
        self.conf_limit = 0.75
        self.hCam, self.wCam = 640, 640

        model_name = 'vgg16_model_4cls_ws_id_2-3_noangle'
        self.detector = htm.handDetector(maxHands=1, detectionCon=0.75)
        self.gesture_model = keras.models.load_model(r'./././model/saved_model/{0}.h5'.format(model_name))
        ###########################################

    def predict(self, img):
        # 모델 추론 및 UI에 영상 출력
        # _, self.cam = self.cpt.read()
        # self.cam = cv2.flip(self.cam, 1)

        # 손 인식시
        img = self.detector.findHands(img)
        self.landmark_list, bbox = self.detector.findPosition(img, draw=False)
        action = ''

        if self.landmark_list:
            self.out_check = 0
            self.fingers = self.detector.fingersUp()
            if 1 not in self.fingers[1:]:
                # 주먹 쥐면 검지의 좌표 저장
                self.in_check += 1
                if self.in_check == 10:
                    self.in_check = 0
                    self.control_mode = True        
            if self.control_mode:
                self.draw_arr.append(self.landmark_list[8][1:])
                cv2.circle(img, tuple(self.landmark_list[8][1:]), 7, (255,0,0), cv2.FILLED)
        else:
            self.out_check += 1
            if self.out_check == 20 and self.control_mode:
                self.control_mode = False
                if not self.control_mode:
                    if 20 < len(self.draw_arr) <= 100:
                        # 저장한 좌표로 input 데이터 생성
                        # 모델 추론
                        self.draw_arr = self.draw_arr[10:-5] # 앞, 뒤 10 frame 씩 제외
                        input_data, imgCanvas = gmm.trans_input(self.draw_arr, self.wCam, self.hCam, self.model_selection)
                        pred, confidence = gmm.predict(self.gesture_model, input_data)
                        
                        if confidence > self.conf_limit:
                            action = AutopyClass.window_controller(pred)
                        # draw.save_file(imgCanvas, self.draw_arr, pred)
                        Canvas = np.zeros((self.wCam, self.hCam, 3), np.uint8)
                    self.draw_arr.clear()

        return self.control_mode, action

                    
# class demopy():
#     def __init__(self):
#         #####################################
#         # webcam size 지정
#         self.hCam, self.wCam = 640, 640
#         # model 설정 값 초기 정의
#         self.model_name = 'vgg16_model_4cls_ws_id_2-3_noangle'
#         self.model_selection = 'CNN'
#         self.conf_limit = 0.75
#         # self.imdraw = dr.imgdraw()
#         # self.imdraw = wb.DryHand()
#         self.gesture_model = keras.models.load_model(r'./././model/saved_model/{0}.h5'.format(self.model_name))
#         self.detector = htm.handDetector(maxHands=1, detectionCon=0.75)
#         self.draw_arr = []
#         self.in_check = 0
#         self.out_check = 0
#         self.control_mode = False
#         #####################################

#     def get_mode(self, img): #, quitbutton=None):
#         # if quitbutton=='q':
#         #     sys.exit()
#         # else:
#         # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         # webcam size 조정
#         # img.set(3, self.wCam)
#         # img.set(4, self.hCam)
#         # _, img = img.read()
#         img = cv2.flip(img, 1)
        
#         # 이미지에서 손인식
#         img = self.detector.findHands(img)
#         landmark_list, _ = self.detector.findPosition(img, draw=False)

#         # 손 인식 되면
#         if landmark_list:
#             self.out_check = 0
#             self.fingers = self.detector.fingersUp()
#             if 1 not in self.fingers[1:]:
#                 # 주먹 쥐면 검지의 좌표 저장
#                 self.in_check += 1
#                 if self.in_check == 10:
#                     self.in_check = 0
#                     self.control_mode = True
#         else:
#             # 손 인식 안되면
#             self.out_check += 1
#             if self.out_check == 10 and self.control_mode:
#                 self.control_mode = False
        
#         return self.control_mode

#             # cv2.imshow('img', img)
#             # cv2.waitKey(0)

#             # self.imdraw.imgdraw(img)

#     def predict(self, img, quitbutton=None):
#         if quitbutton=='q':
#             sys.exit()
#         # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         # webcam size 조정
#         # cap.set(3, self.wCam)
#         # cap.set(4, self.hCam)
#         # _, img = cap.read()
#         img = cv2.flip(img, 1)

        
#         # 이미지에서 손인식
#         img = self.detector.findHands(img)
#         landmark_list, _ = self.detector.findPosition(img, draw=False)

#         # 손 인식 되면
#         if landmark_list:
#             if self.control_mode:
#                 self.draw_arr.append(landmark_list[8][1:])
#                 cv2.circle(img, tuple(landmark_list[8][1:]), 7, (255,0,0), cv2.FILLED)
#         else:
#             # 손 인식 안되면
#             if self.out_check == 10 and self.control_mode:
#                 if not self.control_mode:
#                     if 20 < len(self.draw_arr) <= 100:
#                         # 저장한 좌표로 input 데이터 생성
#                         # 모델 추론
#                         self.draw_arr = self.draw_arr[10:-10] # 앞, 뒤 10 frame 씩 제외
#                         input_data, imgCanvas = gmm.trans_input(self.draw_arr, self.wCam, self.hCam, self.model_selection)
#                         pred, confidence = gmm.predict(self.gesture_model, input_data)
                        
#                         # 예측률 75% 이상 input 데이터 저장
#                         if confidence > self.conf_limit:
#                             AutopyClass.window_controller(pred)
#                             # draw.save_file(imgCanvas, self.draw_arr, pred)
#                         # Canvas = np.zeros((self.wCam, self.hCam, 3), np.uint8)
#                     self.draw_arr.clear()

#         return img