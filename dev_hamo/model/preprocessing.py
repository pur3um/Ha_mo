import dev_hamo.utils.hand_tracking_module as htm
from dev_hamo.utils import draw
import numpy as np
import pandas as pd
import cv2

cam_size = 640

# hand detector class 선언
detector = htm.handDetector(max_hands=1, detection_con=0.75)

cap = cv2.VideoCapture(0)
# 웹 캠의 이미지 조정
cap.set(3, cam_size)
cap.set(4, cam_size)

draw_arr = []
in_check = 0
out_check = 0  # 손이 인식이 안되어 draw_arr가 초기화 되는 것을 막기 위해
control_mode = False  # 좌표 입력 모드 초기화

# 제스처 이동 궤적을 위한 Canvas 초기화
Canvas = np.zeros((cam_size, cam_size, 3), np.uint8) 

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 이미지에서 손 detection
    img = detector.find_hands(img)
    landmark_list, bbox = detector.find_position(img, draw=False)

    # 손이 인식된 경우
    if landmark_list:
        out_check = 0
        fingers = detector.fingers_up()
        # 주먹을 쥐면 좌표 입력 모드 True
        if 1 not in fingers[1:]:
            in_check += 1
            if in_check == 10:
                in_check = 0
                control_mode = True
        if control_mode:
            draw_arr.append(landmark_list[8][1:])
            cv2.circle(img, tuple(landmark_list[8][1:]), 7, (255, 0, 0), cv2.FILLED)
    # 손이 인식 안되면 입력 받은 좌표들로 데이터 생성
    else:
        out_check += 1
        if out_check == 10 and control_mode:
            control_mode = False
            if not control_mode:
                # 최소 20, 최대 100 frame 까지 입력 받기
                if 20 < len(draw_arr) <= 100:
                    draw_arr = draw_arr[10:-10]  # 앞뒤로 10 frame 씩 무시
                    Canvas = draw.draw_canvas(Canvas, len(draw_arr), draw_arr)
                    draw_arr += [[0,0]] * (80 - len(draw_arr))

                    # 데이터 생성
                    draw.save_file(Canvas, draw_arr)
                    Canvas = np.zeros((cam_size, cam_size, 3), np.uint8)  # 캔버스 초기화
                draw_arr.clear()


    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
