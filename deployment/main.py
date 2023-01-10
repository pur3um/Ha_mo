from PyQt5.QtGui import QFont, QFontDatabase
from userInterface import DryHand
from PyQt5.QtWidgets import *
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    fontDB = QFontDatabase()
    fontDB.addApplicationFont('./Bahnschrift.ttf')
    app.setFont(QFont('Bahnschrift'))
    ex = DryHand()
    sys.exit(app.exec_())