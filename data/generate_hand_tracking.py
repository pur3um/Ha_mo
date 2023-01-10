# 구글 비전인식기능 서비스 활용
import mediapipe as mp2
mp = mp2.solutions.mediapipe.python
import cv2

class handDetector():
    """
    Func
    - find_hand : 이미지에서 손을 찾아주는 함수
    - finger_landmark : 찾은 손에서 손가락의 landmark를 반환하는 함수
    - folded_finger : 손가락이 올라왔는지 내려왔는지 구분하는 함수
    """
    def __init__(self, mode=False, max_hands=2, recognition_rate=0.5, tracking_rate=0.5):
        self.mode = mode  # 정적이면 True, 동적이면 False
        self.max_hands = max_hands  # 이미지에서 인식할 손의 최대 개수
        self.recognition_rate = recognition_rate  # 손의 인식률
        self.tracking_rate = tracking_rate  # 손의 추적률

        # mediapipe(mp)의 Hands class를 상속받아 손을 찾아주는 class
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.recognition_rate, self.tracking_rate)

        # 각 손가락의 끝마디의 index
        self.mp_draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hand(self, img, draw=True):
        """
        이미지에서 손을 찾아 각 landmark에 점을 찍고 선을 이어주는 함수
        input parameter
            img : cv2를 통해 입력된 원본 이미지
            draw : landmark를 표시여부
        return
            손가락의 좌표를 그린 img
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # landmark 그리기
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def finger_landmark(self, img, hand_no=0, draw=True):
        """
        손가락들의 landmark의 좌표, bounding box 좌표 반환해주는 함수
        input parameter
            img : 이미지
            hand_no : detection한 손의 index
            draw : bounding box를 그릴지 여부
        반환값
            손가락 좌표를 담은 landmark_list, bounding_box_list
        """
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
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

    def folded_finger(self):
        """
        각 손가락이 펴졌는지 접혀있는지 0과 1로 판단해주는 함수
        반환값
            0과 1을 담은 길이 5의 리스트(0:엄지, 1:검지, 2:중지, 3:약지, 4:새끼)
        """
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