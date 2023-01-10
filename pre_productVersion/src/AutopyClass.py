import pyautogui

control_list = []

def window_controller(pred):
    # 유튜브 모드
    control_list = ['l', 'j', 'k', 'f']
    control_action_list = ['10초 이후', '10초 이전', r'일시정지/재생', r'전체화면/원상복구']
    # j : 10초 이전, l : 10초 이후, k : 일시정지 / 재생, f : 전체화면 / 원상복구
    command = control_list[pred]
    pyautogui.press(command)

    return control_action_list[pred]