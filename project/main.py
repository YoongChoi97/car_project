import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.QtGui import QPixmap
import sqlite3
from PyQt5.QtCore import Qt, QRect

# 데이터베이스 연결
conn = sqlite3.connect('./car.db')
cursor = conn.cursor()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
  
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        self.label.setObjectName("label")
        self.label.setPixmap(QtGui.QPixmap("./image/main.png"))
        
        ### Channel label
        ### 1 Channel
        self.label1_1 = QLabel(self.centralwidget)
        self.label1_1.setGeometry(QtCore.QRect(525, 127, 1357, 932))
        self.label1_1.setObjectName("label1_1")
        self.label1_1.setScaledContents(True)
        
        ### 2 Channel
        self.label2_1 = QLabel(self.centralwidget)
        self.label2_1.setGeometry(QtCore.QRect(525, 127, 685, 932))
        self.label2_1.setObjectName("label2_1")
        self.label2_1.setScaledContents(True)
        
        self.label2_2 = QLabel(self.centralwidget)
        self.label2_2.setGeometry(QtCore.QRect(1261, 127, 685, 932))
        self.label2_2.setObjectName("label2_2")
        self.label2_2.setScaledContents(True)

        ### 4 Channel
        self.label3_1 = QLabel(self.centralwidget)
        self.label3_1.setGeometry(QtCore.QRect(600, 160, 558, 330))
        self.label3_1.setObjectName("label3_1")
        self.label3_1.setScaledContents(True)
        
        self.label3_2 = QLabel(self.centralwidget)
        self.label3_2.setGeometry(QtCore.QRect(1247, 160, 558, 330))
        self.label3_2.setObjectName("label3_2")
        self.label3_2.setScaledContents(True)
        
        self.label3_3 = QLabel(self.centralwidget)
        self.label3_3.setGeometry(QtCore.QRect(600, 548, 558, 330))
        self.label3_3.setObjectName("label3_3")
        self.label3_3.setScaledContents(True)
        
        self.label3_4 = QLabel(self.centralwidget)
        self.label3_4.setGeometry(QtCore.QRect(1247, 548, 558, 330))
        self.label3_4.setObjectName("label3_4")
        self.label3_4.setScaledContents(True)

        # 드롭박스 추가
        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(932, 57, 287, 37))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Main Channel")
        self.comboBox.addItem("1 Channel")
        self.comboBox.addItem("2 Channel")
        self.comboBox.addItem("4 Channel")
        self.comboBox.setStyleSheet("font: 75 9pt 'Nirmala UI';")
        self.comboBox.activated[str].connect(self.open_channel)
        self.comboBox.currentIndexChanged.connect(self.open_channel)
        #self.comboBox.setStyleSheet("background: transparent; border: none;")
        
          
        self.query_button = QPushButton(self.centralwidget)
        self.query_button.setGeometry(172, 234, 185, 30)
        self.query_button.setObjectName("PushButton")
        # self.query_button.setStyleSheet("background: transparent; border: none;")

        self.query_by_name_button = QPushButton(self.centralwidget)
        self.query_by_name_button.setGeometry(109, 419, 200, 30)
        self.query_by_name_button.setObjectName("PushButton")
        # self.query_by_name_button.setStyleSheet("background: transparent; border: none;")

        self.query_by_name2_button = QPushButton(self.centralwidget)
        self.query_by_name2_button.setGeometry(109, 498, 200, 30)
        self.query_by_name2_button.setObjectName("PushButton")
        # self.query_by_name2_button.setStyleSheet("background: transparent; border: none;")



        self.data_display = QTableWidget(self.centralwidget)
        self.data_display.setStyleSheet("QTableWidget { border-radius: 20px; }")
        self.data_display.setGeometry(106, 575, 358, 350)
        self.data_display.setColumnCount(4)
        self.data_display.setHorizontalHeaderLabels(['id','car_plate', 'file_name', 'upload_time'])

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def open_channel(self): 
        text = self.comboBox.currentText()
        if text == "Main Channel" :
            self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
            # self.label.setPixmap(QtGui.QPixmap("./image/main1.png"))
            self.label.setPixmap(QtGui.QPixmap("./image/main.png"))
            
            self.label1_1.setGeometry(QtCore.QRect(800, 259, 772, 493))
            
        elif text == "1 Channel" :
            self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
            # self.label.setPixmap(QtGui.QPixmap("./image/main1.png"))
            self.label.setPixmap(QtGui.QPixmap("./image/main1.png"))
            
            self.label1_1.setGeometry(QtCore.QRect(800, 259, 772, 493))
            #self.label1_1.setStyleSheet("color: blue; background-color: red;")
            self.label1_1.setHidden(False)
            self.label2_1.setHidden(True)
            self.label2_2.setHidden(True)
            self.label3_1.setHidden(True)
            self.label3_2.setHidden(True)
            self.label3_3.setHidden(True)
            self.label3_4.setHidden(True)
        elif text == "2 Channel" :
            self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
            self.label.setPixmap(QtGui.QPixmap("./image/main2.png"))
            self.label2_1.setGeometry(QtCore.QRect(600, 326, 558, 330))
            #self.label2_1.setStyleSheet("color: blue; background-color: red;")
            self.label2_2.setGeometry(QtCore.QRect(1248, 323, 558, 330))
            #self.label2_2.setStyleSheet("color: blue; background-color: red;")
            self.label1_1.setHidden(True)
            self.label2_1.setHidden(False)
            self.label2_2.setHidden(False)
            self.label3_1.setHidden(True)
            self.label3_2.setHidden(True)
            self.label3_3.setHidden(True)
            self.label3_4.setHidden(True)
        elif text == "4 Channel" :
            self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
            self.label.setPixmap(QtGui.QPixmap("./image/main4.png"))
            self.label3_1.setGeometry(QtCore.QRect(600, 160, 558, 330))
            #self.label3_1.setStyleSheet("color: blue; background-color: red;;")
            self.label3_2.setGeometry(QtCore.QRect(1247, 160, 558, 330))
            #self.label3_2.setStyleSheet("color: blue; background-color: red;")
            self.label3_3.setGeometry(QtCore.QRect(600, 548, 558, 330))
            #self.label3_3.setStyleSheet("color: blue; background-color: red;")
            self.label3_4.setGeometry(QtCore.QRect(1247, 548, 558, 330))
            #self.label3_4.setStyleSheet("color: blue; background-color: red;")
            self.label1_1.setHidden(True)
            self.label2_1.setHidden(True)
            self.label2_2.setHidden(True)
            self.label3_1.setHidden(False)
            self.label3_2.setHidden(False)
            self.label3_3.setHidden(False)
            self.label3_4.setHidden(False)    

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.query_button.clicked.connect(self.get_all_data)
        self.ui.query_by_name_button.clicked.connect(lambda: self.get_data_by_name(0, 4))
        self.ui.query_by_name2_button.clicked.connect(lambda: self.get_data_by_name2(4, 8))
        
        # 모든데이터, 앞자리, 뒷자리 표시
        self.ui.query_button.setGeometry(QRect(172, 234, 188, 30))
        self.ui.query_button.setStyleSheet("background: transparent; border: none;")
        self.ui.query_by_name_button.setGeometry(QRect(172, 323, 188, 30))
        self.ui.query_by_name_button.setStyleSheet("background: transparent; border: none;")
        self.ui.query_by_name2_button.setGeometry(QRect(172, 361, 188, 30))
        self.ui.query_by_name2_button.setStyleSheet("background: transparent; border: none;")


        self.ui.data_display.cellClicked.connect(self.show_image)
        self.ui.data_display.itemSelectionChanged.connect(self.show_image)
        
        # 여러 개의 채널 라벨 추가
        self.labels = [self.ui.label1_1, self.ui.label2_1, self.ui.label2_2, self.ui.label3_1, self.ui.label3_2, self.ui.label3_3, self.ui.label3_4]

        self.clear_button = QPushButton(self)
        self.clear_button.setGeometry(QRect(172, 480, 188, 30))
        self.clear_button.setStyleSheet("background: transparent; border: none;")
        self.clear_button.clicked.connect(self.clear_labels)
        
    def get_all_data(self):
        cursor.execute("SELECT id, car_plate, file_name, upload_time FROM car_Table")
        data = cursor.fetchall()
        self.display_data(data)

    def get_data_by_name(self, start, end):
        name, ok = QInputDialog.getText(self, "앞자리 입력", "한글 기준 앞 2~3자리를 입력하세요:", QLineEdit.Normal, "")
        if ok and name:
            cursor.execute(f"SELECT id, car_plate, file_name, upload_time FROM car_Table WHERE substr(car_plate, {start}, {end}) LIKE ?", (name + '%',))
            data = cursor.fetchall()
            self.display_data(data)

    def get_data_by_name2(self, start, end):
        name, ok = QInputDialog.getText(self, "뒷자리 입력", "한글 기준 뒤 4자리를 입력하세요:", QLineEdit.Normal, "")
        if ok and name:
            cursor.execute(f"SELECT id, car_plate, file_name, upload_time FROM car_Table WHERE car_plate LIKE ?", ('%' + name,))
            data = cursor.fetchall()
            self.display_data(data)

    def display_data(self, data):
        self.ui.data_display.setRowCount(len(data))
        for row_index, row in enumerate(data):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.ui.data_display.setItem(row_index, column_index, item)

    def show_image(self):
        selected_items = self.ui.data_display.selectedItems()
        if not selected_items:
            return
        selected_row = selected_items[0].row()
        file_name = self.ui.data_display.item(selected_row, 2).text()
        self.display_image(file_name)

    def display_image(self, file_name):
        file_path = 'car_save_image/' + file_name
        if os.path.isfile(file_path):
            image = QPixmap(file_path)
            if not image.isNull():
                for label in self.labels:
                    if label.isVisible() and label.pixmap() is None:
                        label.setPixmap(image.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
                        break
            else:
                QMessageBox.warning(self, "파일 열기 오류", "파일을 열 수 없습니다. 파일이 손상되었거나, 지원하지 않는 형식입니다.")
        else:
            QMessageBox.warning(self, "파일이 없음", "해당 파일을 찾을 수 없습니다.")

    def clear_labels(self):
        for label in self.labels:
            label.clear()
            label.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())