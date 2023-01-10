import mediapipe as mp2
mp = mp2.solutions.mediapipe.python
import cv2

class HandDetector():
    '''
    함수 정리
        findHands : 이미지에서 손을 찾아주는 함수
        findPosition : 찾은 손에서 손가락의 landmark를 반환하는 함수
        fingersUp : 손가락이 올라왔는지 내려왔는지 구분하는 함수
    '''
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        # mediapipe의 hands class를 상속받음 
        self.mode = mode # 정적이면 True, 동적이면 False
        self.maxHands = maxHands # 이미지에서 인식할 손의 최대 개수
        self.detectionCon = detectionCon # 손의 인식률
        self.trackingCon = trackingCon # 손의 추적률

        self.mp_hands = mp.solutions.hands 
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackingCon)
        # mediapipe의 hands class를 상속받아 손을 찾아주는 class

        self.mp_draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] # 각 손가락의 끝마디의 index

    def findHands(self, img, draw = True):
        '''
        이미지에서 손을 찾아 각 landmark에 점을 찍고 선을 이어주는 함수
        input parameter
            img : cv2를 통해 입력된 원본 이미지
            draw : landmark를 표시여부
        반환값
            손가락의 좌표를 그린 img
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # landmark 그리기
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        '''
        손가락들의 landmark의 좌표, bounding box 좌표 반환해주는 함수
        input parameter
            img : 이미지
            handNo : detection한 손의 index
            draw : bounding box를 그릴지 여부 
        반환값
            손가락 좌표를 담은 landmark_list, boundingbox_list
        '''
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # 화면의 비율을 추출하여 landmarks들의 위치를 찾는다.
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        '''
        각 손가락이 펴졌는지 접혀있는지 0과 1로 판단해주는 함수
        반환값
            0과 1을 담은 길이 5의 리스트(0:엄지, 1:검지, 2:중지, 3:약지, 4:새끼)
        '''
        fingers = []
        # Thumb[4, 8, 12, 16, 20]
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers