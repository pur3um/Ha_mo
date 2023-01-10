import cv2

def draw_canvas(img, frame_cnt, draw_arr):
    '''
    전달 받은 좌표 arr로 이미지 데이터 생성
    input_parameter:
        img: opencv에서 받아온 이미지
        frame_cnt: 동영상frame 개수
        draw_arr: 검지의 좌표 list
    반환값:
        img: 캔버스에 궤적을 그린 이미지
    '''
    prev_index = tuple(draw_arr[0])
    for i in range(1, frame_cnt):
        # 이전, 현재 좌표 저장하여 선 연결
        curr_index = tuple(draw_arr[i])
        cv2.line(img, prev_index, curr_index, (255, 255, 255), 7)
        prev_index = curr_index
    return img