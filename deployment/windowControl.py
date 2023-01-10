import pyautogui
import pywinauto
import pygetwindow as gw
import time

class WindowController():

    def youtube(self, pred):
        # 유튜브 모드
        control_list = ['l', 'j', 'k', 'f', ['ctrl','l'], '', ['shift','n'], ['alt','left']]
        control_action_list = ['10초 이후', '10초 이전', '일시정지/재생', '전체화면/원상복구',
                            '유튜브 켜기', '광고스킵', '다음영상', '이전페이지']
        # j : 10초 이전, l : 10초 이후, k : 일시정지 / 재생, f : 전체화면 / 원상복구
        if pred == control_action_list.index('유튜브 켜기'):
            if gw.getWindowsWithTitle('Chrome'): # 윈도우 타이틀에 Chrome 이 포함된 모든 윈도우 수집, 리스트로 리턴
                win = gw.getWindowsWithTitle('Chrome')[0]
                if win.isActive == False:
                    pywinauto.application.Application().connect(handle=win._hWnd).top_window().set_focus()
                    win.activate() #윈도우 활성화
            else:
                pyautogui.press('win')
                pyautogui.typewrite('chrome')
                pyautogui.press('enter')
                time.sleep(1)

            pyautogui.hotkey(control_list[pred][0], control_list[pred][1])
            pyautogui.typewrite('www.youtube.com')
            pyautogui.press('enter')
        
        elif pred == control_action_list.index('광고스킵'):
            point_to_click = pyautogui.locateCenterOnScreen('images/skip_btn.PNG',confidence=0.5)
            if point_to_click == None:
                point_to_click = pyautogui.locateCenterOnScreen('images/skipadd.PNG',confidence=0.5)
            if point_to_click:
                pyautogui.click(point_to_click)
        
        elif pred == control_action_list.index('다음영상')\
            or pred == control_action_list.index('이전페이지'):
            pyautogui.hotkey(control_list[pred][0], control_list[pred][1])
        
        else:
            command = control_list[pred]
            pyautogui.press(command)

        return control_action_list[pred]

    def webMode(self, pred):
        if pred < 4:
            control_list = [['alt','right'], ['alt','left'],['win','chrome','enter'], ['alt','f4']]
            control_action_list = ['다음', '이전', '크롬창켜기', '크롬창끄기']
        else:
            return "명령어가 없습니다."

        if pred == control_action_list.index('크롬창켜기'):
            if gw.getWindowsWithTitle('Chrome'): # 윈도우 타이틀에 Chrome 이 포함된 모든 윈도우 수집, 리스트로 리턴
                win = gw.getWindowsWithTitle('Chrome')[0]
                if win.isActive == False:
                    pywinauto.application.Application().connect(handle=win._hWnd).top_window().set_focus()
                    win.activate() #윈도우 활성화
            else:
                pyautogui.press(control_list[pred][0])
                time.sleep(0.1)
                pyautogui.typewrite(control_list[pred][1])
                time.sleep(0.1)
                pyautogui.press(control_list[pred][2])
                time.sleep(1) 
        else:
            pyautogui.hotkey(control_list[pred][0], control_list[pred][1])
        return control_action_list[pred]


    def presentMode(self, pred):
        if pred < 4:
            control_list = [['right',""], ['left',""],['ctrl','f5'], ['esc',""]]
            control_action_list = ['다음', '이전', '전체창', '전체창끄기']
            pyautogui.hotkey(control_list[pred][0], control_list[pred][1])
            return control_action_list[pred]
        return "명령어가 없습니다."