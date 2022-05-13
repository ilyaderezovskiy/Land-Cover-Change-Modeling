from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets


# Класс страницы информационного уведомления пользователя
class InfoWindow(QDialog):
    # Конструктор класса
    def __init__(self, text="", title="", parent=None):
        super(InfoWindow, self).__init__(parent)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self)
        self.label.setText(text)
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.close_form)
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton.setText("OK")
        self.setWindowTitle(title)

    # Метод добавления текста сообщения на форму
    def set_text(self, text):
        self.label.setText(text)

    # Метод закрытия формы
    def close_form(self):
        self.close()
