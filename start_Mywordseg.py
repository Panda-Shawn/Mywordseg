from os import error
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from main import inference
from Ui_Mywordseg import Ui_MainWindow


class ControlBoard(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ControlBoard, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.FORWARD)
        self.pushButton_2.clicked.connect(self.BACKWARD)
        self.pushButton_3.clicked.connect(self.BI_DIRECTION)
        self.pushButton_4.clicked.connect(self.HMM)
        self.pushButton_5.clicked.connect(self.BI_DIRECTION)

    def FORWARD(self):
        try:
            input = self.textEdit.toPlainText()
            output = inference(input, 'forward')
            self.textBrowser.setText(output)
        except:
            self.textBrowser.setText("请输入正确的语句。")

    def BACKWARD(self):
        try:
            input = self.textEdit.toPlainText()
            output = inference(input, 'backward')
            self.textBrowser.setText(output)
        except:
            self.textBrowser.setText("请输入正确的语句。")

    def BI_DIRECTION(self):
        try:
            input = self.textEdit.toPlainText()
            output = inference(input, 'bi-direction')
            self.textBrowser.setText(output)
        except:
            self.textBrowser.setText("请输入正确的语句。")

    def HMM(self):
        input = self.textEdit.toPlainText()
        print(input)
        output = inference(input, 'hmm')
        self.textBrowser.setText(output)
        # try:
        #     input = self.textEdit.toPlainText()
        #     output = inference(input, 'hmm')
        #     self.textBrowser.setText(output)
        # except:
        #     self.textBrowser.setText("请输入正确的语句。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ControlBoard()
    win.show()
    sys.exit(app.exec_())

