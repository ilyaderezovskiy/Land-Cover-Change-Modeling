from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets


# Класс страницы подтверждения пользователем выполняемого действия
class AgreementWindow(QDialog):
    # Конструктор класса
    def __init__(self, parent=None):
        super(AgreementWindow, self).__init__(parent)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.text_label = QtWidgets.QLabel(self)
        self.text_label.setText("Вы уверены, что хотите удалить изображение?")
        self.agreeButton = QtWidgets.QPushButton(self)
        self.agreeButton.setObjectName("agreeButton")
        self.agreeButton.clicked.connect(self.agree)
        self.disagreeButton = QtWidgets.QPushButton(self)
        self.disagreeButton.setObjectName("disagreeButton")
        self.disagreeButton.clicked.connect(self.disagree)
        self.verticalLayout.addWidget(self.agreeButton)
        self.verticalLayout.addWidget(self.disagreeButton)
        self.verticalLayout.addWidget(self.text_label)
        self.verticalLayout.addWidget(self.agreeButton)
        self.verticalLayout.addWidget(self.disagreeButton)
        self.agreeButton.setText("YES")
        self.disagreeButton.setText("NO")
        self.setWindowTitle("Подтверждение удаления")

    # Метод положительного ответа пользователя о вносимых изменения
    def agree(self):
        self.result = True
        self.close()

    # Метод отрицательного ответа пользователя о вносимых изменения
    def disagree(self):
        self.result = False
        self.close()

    # Метод получение результата ответа пользователя о вносимых изменения
    def getResult(self):
        return self.result
