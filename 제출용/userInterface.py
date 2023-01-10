from re import S
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import getGesture
import cv2
import sys

class DryHand(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ha-mo')
        self.setGeometry(150, 150, 900, 600)
        self.initUI()
    
    def initUI(self):
        self.timer = QTimer()
        self.cpt = cv2.VideoCapture(0)
        self.dp = getGesture.Gesture()
        self.fps = 40
        self.action_label = ''
        # 화면 관련 설정
        self.bright = 10
        self.win_width = 900
        self.win_height = 600
        self.frame_width = 640
        self.frame_height = 480
        self.btn_width = int((self.win_width-260)//2)
        self.btn_height = 40
        self.btn1_width = int((self.win_width-260)//3)
        self.btn1_height = 40
        self.center_labels = 75
        self.modeChange = 0

        self.frame = QLabel(self)
        font1 = self.frame.font()
        font1.setPointSize(20)
        self.frame.setAlignment(Qt.AlignCenter)
        self.frame.resize(self.frame_width, self.frame_height)
        self.frame.setScaledContents(True)
        self.frame.move(5,5)

        # 손동작 이미지 출력
        self.frame1 = QLabel(self)
        self.frame1.resize(200, 150)
        self.frame1.setStyleSheet("color: #FF5733; border-style: solid; border-width: 1px; border-color: #05F1F5; border-radius: 10px; ")
        self.frame1.setScaledContents(True)
        self.frame1.move(self.frame_width + 30 , 60)

        self.btn_on = QPushButton("Play Cam", self)
        self.btn_on.resize(self.btn_width, self.btn_height)
        self.btn_on.setStyleSheet('color: #FCFAFF; background: #159282')
        self.btn_on.move(5, self.frame_height + 10)
        self.btn_on.clicked.connect(self.start)

        self.btn_off = QPushButton("Stop Cam", self)
        self.btn_off.resize(self.btn_width, self.btn_height)
        self.btn_off.setStyleSheet('color: #FCFAFF; background: #159282')
        self.btn_off.move(5 + self.btn_width + 5, self.frame_height + 10)
        self.btn_off.clicked.connect(self.stop)

        self.btn_mode1 = QPushButton("Youtube Mode", self)
        self.btn_mode1.resize(self.btn1_width, self.btn1_height)
        self.btn_mode1.setStyleSheet('color: #FCFAFF; background: #159282')
        self.btn_mode1.move(5, self.frame_height + 60)
        self.btn_mode1.clicked.connect(lambda: self.change_mode(0))

        self.btn_mode2 = QPushButton("Web Mode", self)
        self.btn_mode2.resize(self.btn1_width, self.btn1_height)
        self.btn_mode2.setStyleSheet('color: #FCFAFF; background: #159282')
        self.btn_mode2.move(5 + self.btn1_width+2.5, self.frame_height + 60)
        self.btn_mode2.clicked.connect(lambda: self.change_mode(1))

        self.btn_mode3 = QPushButton("Presentation Mode", self)
        self.btn_mode3.resize(self.btn1_width, self.btn1_height)
        self.btn_mode3.setStyleSheet('color: #FCFAFF; background: #159282')
        self.btn_mode3.move(5 + (self.btn1_width*2)+5, self.frame_height + 60)
        self.btn_mode3.clicked.connect(lambda: self.change_mode(2))

        self.prt = QLabel(self)
        font2 = self.prt.font()
        font2.setPointSize(20)
        self.prt.setStyleSheet("color: #FF5733; border-style: solid; border-width: 1px; border-color: #05F1F5; border-radius: 10px; ")
        self.prt.setAlignment(Qt.AlignCenter)
        self.prt.resize(220, self.btn_height)
        self.prt.move(self.frame_width + (self.win_width-self.frame_width)//2 - self.center_labels - 40, self.frame_height + 10) # 5 +  (self.btn_width + 5)*2

        self.sldr = QSlider(Qt.Horizontal, self)
        self.sldr.resize(100, 25)
        self.sldr.move(self.win_width-105-105-30, self.frame_height + 70)
        self.sldr.setMinimum(1)
        self.sldr.setMaximum(60)
        self.sldr.setValue(40)
        self.sldr.valueChanged.connect(self.setFps)

        self.sldr1 = QSlider(Qt.Horizontal, self)
        self.sldr1.resize(100, 25)
        self.sldr1.move(self.win_width-105-10, self.frame_height + 70)
        self.sldr1.setMinimum(-100)
        self.sldr1.setMaximum(100)
        self.sldr1.setValue(0)
        self.sldr1.valueChanged.connect(self.setBright)
        
        # 모드와 현재 동작 표기
        self.mode1 = QLabel("Current Mode", self)
        font3 = self.mode1.font()
        font3.setPointSize(20)
        self.mode1.setStyleSheet("color: #000000;border-style: solid; border-width: 3px; border-color: #96FF8D; border-radius: 10px; ")
        self.mode1.setAlignment(Qt.AlignCenter)
        self.mode1.resize(150, 25)
        self.mode1.move(self.frame_width + (self.win_width-self.frame_width)//2 - self.center_labels, 250)
        
        self.mode2 = QLabel(self)
        font4 = self.mode2.font()
        font4.setPointSize(20)
        self.mode2.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #96FF8D; border-radius: 10px; ")
        self.mode2.setAlignment(Qt.AlignCenter)
        self.mode2.resize(100, 25)
        self.mode2.move(self.win_width-105-105-30, 280)
        
        self.mode3 = QLabel(self)
        font5 = self.mode3.font()
        font5.setPointSize(20)
        self.mode3.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #96FF8D; border-radius: 10px; ")
        self.mode3.setAlignment(Qt.AlignCenter)
        self.mode3.resize(100, 25)
        self.mode3.move(self.win_width-105-10, 280)
        
        self.act0 = QLabel("Last action Image", self)
        font6 = self.act0.font()
        font6.setPointSize(20)
        self.act0.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #07DE6D; border-radius: 10px; ")
        self.act0.setAlignment(Qt.AlignCenter)
        self.act0.resize(150, 25)
        self.act0.move(self.frame_width + (self.win_width-self.frame_width)//2 - self.center_labels, 20)

        self.act1 = QLabel("Last action", self)
        font6 = self.act1.font()
        font6.setPointSize(20)
        self.act1.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #07DE6D; border-radius: 10px; ")
        self.act1.setAlignment(Qt.AlignCenter)
        self.act1.resize(150, 25)
        self.act1.move(self.frame_width + (self.win_width-self.frame_width)//2 - self.center_labels, 340)
        
        self.act2 = QLabel(self)
        font7 = self.act2.font()
        font7.setPointSize(20)
        self.act2.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #07DE6D; border-radius: 10px; ")
        self.act2.setAlignment(Qt.AlignCenter)
        self.act2.resize(150, 25)
        self.act2.move(self.frame_width + (self.win_width-self.frame_width)//2 - self.center_labels, 370)

        self.act3 = QLabel("프레임 조절", self)
        font6.setPointSize(11)
        self.act3.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #07DE6D; border-radius: 7px; ")
        self.act3.setAlignment(Qt.AlignCenter)
        self.act3.resize(100, 25)
        self.act3.move(self.win_width-105-105-30, 430)

        self.act4 = QLabel("명암 조절",self)
        font6.setPointSize(11)
        self.act4.setStyleSheet("color: #000000; border-style: solid; border-width: 3px; border-color: #07DE6D; border-radius: 7px; ")
        self.act4.setAlignment(Qt.AlignCenter)
        self.act4.resize(100, 25)
        self.act4.move(self.win_width-105-10, 430)

        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', '종료 하시겠습니까?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.cont = False
            event.accept()
        else:
            event.ignore()

    def setFps(self):
        self.fps = self.sldr.value()
        self.prt.setText("Set FPS to " + str(self.fps))
        self.timer.stop()
        self.timer.start(1000. / self.fps)

    def setBright(self):
        self.bright = self.sldr1.value()
        self.prt.setText("Set brightness to " + str(self.bright))
        self.timer.stop()
        self.timer.start(1000. / self.fps)

    def start(self):
        self.timer.timeout.connect(self.predict)
        self.timer.start(1000 / self.fps)

    def stop(self):
        self.frame.setPixmap(QPixmap.fromImage(QImage()))
        self.timer.stop()

    def predict(self):
        _, self.cam = self.cpt.read()
        self.cam = cv2.flip(self.cam, 1)
        self.cam = cv2.add(self.cam, (self.bright,self.bright,self.bright,0))
        self.control_mode, tmp_action_label, Canvas_img = self.dp.predict(self.cam, self.modeChange)

        if self.control_mode:
            self.mode2.setText("Action Input")
            self.mode3.setText("")
        else:
            self.mode2.setText("")
            self.mode3.setText("No Edit")

        if tmp_action_label:
            self.action_label = tmp_action_label
        self.act2.setText(self.action_label)
        
        # GUI에 이미지 출력
        self.cam = cv2.cvtColor(self.cam, cv2.COLOR_BGR2RGB)
        img = QImage(self.cam, self.cam.shape[1], self.cam.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.frame.setPixmap(pix)

        if Canvas_img is not None:
            img_canvas = Canvas_img
            img_canvas = QImage(img_canvas, img_canvas.shape[1], img_canvas.shape[0], QImage.Format_RGB888)
            img_canvas = QPixmap.fromImage(img_canvas)
            self.frame1.setPixmap(img_canvas)
    
    def change_mode(self, mode):
        self.modeChange = mode
