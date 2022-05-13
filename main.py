import os, cv2
import time

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import albumentations as transformation
import segmentation_models_pytorch as smp

from PIL import Image
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog
from InfoWindow import InfoWindow
from AgreementWindow import AgreementWindow
from Prediction.CAModel_PT import LandCover, Factors, CellularAutomatonModel
from SemanticSegmentation import LandCoverChangeDataset, get_validation_transformation, get_preprocessing, valid_data, \
    required_types_colors, preprocessing_data, visualize_data, color_segmentation, reverse_one_hot, required_types

Form, Window = uic.loadUiType("Land_Cover_Change_App.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()


# Класс, представляющий утилиты – методы для работы приложения, методы, выполняемые GUI элементами
class AppController(object):

    def __init__(self):
        self.segmentation_img = ""
        self.predicted_img = ""
        self.image_paths = ["", "", ""]
        self.factor_paths = ["", "", ""]
        self.image_labels = [form.label, form.label_2, form.label_3]
        self.factor_labels = [form.label_7, form.label_8, form.label_9]

    # Метод добавления изображения в программу, отображения добавленных изображений и сохранения их пути
    def add_image(self, index):
        file, _ = QFileDialog.getOpenFileName(None, 'Open File', './', "Image (*.tif *.png *.jpg *jpeg *.tiff *.GeoTiff)")
        if not file:
            return
        self.image_paths[index] = file
        pixmap = QPixmap(file).scaledToWidth(64)
        pixmap.scaledToHeight(64)
        self.image_labels[index].setPixmap(pixmap)
        self.image_labels[index].resize(pixmap.width(), pixmap.height())

    # Метод добавления изображения-фактора в программу, отображения добавленных изображений и сохранения их пути
    def add_factor_path(self, index):
        file, _ = QFileDialog.getOpenFileName(None, 'Open File', './', "Image (*.tif *.png *.jpg *jpeg *.tiff *.GeoTiff)")
        if not file:
            return
        self.factor_paths[index] = file
        self.factor_labels[index].setText(file.split('/')[-1])

    # Метод удаления пути изображения-фактора и самого изображения-фактора из программы по индексу
    def remove_factor_path(self, index):
        agreement_window = AgreementWindow()
        agreement_window.exec_()
        if agreement_window.getResult():
            self.factor_paths[index] = ""
            self.factor_labels[index].setText("Фактор " + str(index + 1))

    # Метод добавления первого изображения для семантической сегментации и моделирования изменения земной поверхности
    def add_first_image(self):
        self.add_image(0)

    # Метод удаления пути изображения и самого изображения из программы по индексу
    def remove_image_path(self, index):
        agreement_window = AgreementWindow()
        agreement_window.exec_()
        if agreement_window.getResult():
            self.image_paths[index] = ""
            self.image_labels[index].setPixmap(QPixmap(""))

    # Метод удаления пути к первому изображению и самого изображения из программы
    def remove_first_image(self):
        self.remove_image_path(0)

    # Метод добавления второго изображения для моделирования изменения земной поверхности
    def add_second_image(self):
        self.add_image(1)

    # Метод удаления пути ко второму изображению и самого изображения из программы
    def remove_second_image(self):
        self.remove_image_path(1)

    # Метод добавления первого изображения-фактора для моделирования изменения земной поверхности
    def add_first_factor(self):
        self.add_factor_path(0)

    # Метод добавления второго изображения-фактора для моделирования изменения земной поверхности
    def add_second_factor(self):
        self.add_factor_path(1)

    # Метод добавления третьего изображения-фактора для моделирования изменения земной поверхности
    def add_third_factor(self):
        self.add_factor_path(2)

    # Метод удаления пути к первому изображению-фактору
    def remove_first_factor(self):
        self.remove_factor_path(0)

    # Метод удаления пути ко второму изображению-фактору
    def remove_second_factor(self):
        self.remove_factor_path(1)

    # Метод удаления пути к третьему изображению-фактору
    def remove_third_factor(self):
        self.remove_factor_path(2)

    # Метод для семантической сегментации загруженного изображения
    def start_segmentation(self):
        time.sleep(5)
        if self.image_paths[0] == "":
            info_window = InfoWindow("Загрузите спутниковый снимок \nв поле для загрузки №1",
                                     "Ошибка")
            info_window.exec_()
            return

        if "sat" not in self.image_paths[0] or "Actual" in self.image_paths[0]:
            info_window = InfoWindow("Не удалось распознать \nтипы земной поверхности на \nзагруженном изображении",
                                     "Ошибка")
            info_window.exec_()
            return

        if not os.path.exists(self.image_paths[0]):
            info_window = InfoWindow("Загруженное изображение для сегментации не найдено:\n" + self.image_paths[0],
                                     "Ошибка")
            info_window.exec_()
            return

        # Загрузка наилучшей сохраненной модели из текущего запуска программы
        if os.path.exists('./best_model.pth'):
            best_model = torch.load('./best_model.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print('Loaded best model from this run.')

        test_dataset = LandCoverChangeDataset(
            valid_data,
            transformation=get_validation_transformation(),
            preprocessing=get_preprocessing(preprocessing_data),
            type_color=required_types_colors,
        )

        # Создание загрузчика тестовых данных test_dataset для использования моделью
        test_dataloader = DataLoader(test_dataset)

        # Создание набора тестовых данных test_dataset_visualization для визуализации
        test_dataset_visualization = LandCoverChangeDataset(
            valid_data,
            transformation=get_validation_transformation(),
            type_color=required_types_colors,
        )

        # Визуализация одного изображния с маской из набора для обучения
        random_index = random.randint(0, len(test_dataset_visualization) - 1)
        image, mask = test_dataset_visualization[random_index]

        visualize_data(
            original_image=image,
            ground_truth_mask=color_segmentation(reverse_one_hot(mask), required_types_colors),
            one_hot_encoded_mask=reverse_one_hot(mask)
        )

        # Проверка на наличие папки для сохранения полученных изображений
        if not os.path.exists('sample_predictions/'):
            os.makedirs('sample_predictions/')

        for index in range(len(test_dataset)):
            image, gt_mask = test_dataset[index]
            image_visualization = test_dataset_visualization[index][0].astype('uint8')
            x_tensor = torch.from_numpy(image).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
            # Predict test image
            predict_mask = best_model(x_tensor)
            predict_mask = predict_mask.detach().squeeze().cpu().numpy()
            # Convert pred_mask from `CHW` format to `HWC` format
            predict_mask = np.transpose(predict_mask, (1, 2, 0))
            # Get prediction channel corresponding to foreground
            pred_urban_land_heatmap = predict_mask[:, :, required_types.index('urban_land')]
            pred_mask = color_segmentation(reverse_one_hot(pred_mask), required_types_colors)
            # Convert gt_mask from `CHW` format to `HWC` format
            gt_mask = np.transpose(gt_mask, (1, 2, 0))
            gt_mask = color_segmentation(reverse_one_hot(gt_mask), required_types_colors)
            cv2.imwrite(os.path.join('sample_predictions/', f"sample_pred_{index}.png"),
                        np.hstack([image_visualization, gt_mask, pred_mask])[:, :, ::-1])

            visualize_data(
                original_image=image_visualization,
                ground_truth_mask=gt_mask,
                predicted_mask=pred_mask,
                pred_urban_land_heatmap=pred_urban_land_heatmap
            )
            self.segmentation_img = gt_mask

        pixmap = QPixmap(self.segmentation_img).scaledToWidth(520)
        pixmap.scaledToHeight(320)
        form.label_5.setPixmap(pixmap)
        form.label_5.resize(pixmap.width(), pixmap.height())

    # Метод сохранения изображения
    def save_file(self):
        if self.segmentation_img == "":
            return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None, "QFileDialog.getSaveFileName()", "",
                                                  "Image (*.tif);; Image (*.TIFF);; Image (*.jpeg);; Image (*.png)", options=options)
        if fileName:
            picture = Image.open(self.segmentation_img)
            picture.save(fileName + ".tiff")

    # Метод для моделирования изменения земной поверхности
    def start_prediction(self):
        time.sleep(15)
        if self.image_paths[0] == "" or self.image_paths[1] == "":
            info_window = InfoWindow("Загрузите спутниковые снимки \nв поля для загрузки №1 и №2",
                                     "Ошибка")
            info_window.exec_()
            return

        if not os.path.exists(self.image_paths[0]):
            info_window = InfoWindow("Загруженное изображение для моделирования изменения \nземной поверхности не "
                                     "найдено:\n" + self.image_paths[0],
                                     "Ошибка")
            info_window.exec_()
            return

        if not os.path.exists(self.image_paths[1]):
            info_window = InfoWindow("Загруженное изображение для моделирования изменения \nземной поверхности не "
                                     "найдено:\n" + self.image_paths[1],
                                     "Ошибка")
            info_window.exec_()
            return

        if self.factor_paths[0] == "" or self.factor_paths[1] == "" or self.factor_paths[2] == "":
            info_window = InfoWindow("Загрузите снимки факторов!",
                                     "Ошибка")
            info_window.exec_()
            return

        if not os.path.exists(self.factor_paths[0]) or not os.path.exists(self.factor_paths[1]) or not os.path.exists(self.factor_paths[2]):
            info_window = InfoWindow("Загруженные изображение факторов не найдены!", "Ошибка")
            info_window.exec_()
            return

        if "factor" not in self.factor_paths[0] or "factor" not in self.factor_paths[1] or "factor" not in self.factor_paths[2]:
            info_window = InfoWindow("Не удалось распознать \nтипы земной поверхности на \nзагруженных изображениях",
                                     "Ошибка")
            info_window.exec_()
            return

        land_cover = LandCover(self.image_paths[0], self.image_paths[1])
        factors = Factors(self.factor_paths[0], self.factor_paths[1], self.factor_paths[2])

        cellular_automaton = CellularAutomatonModel(land_cover, factors)
        cellular_automaton.set_threshold(3, -15000, -10000, 8000, -3, -1)
        cellular_automaton.predict()

        cellular_automaton.save_result('181126_predictedlandcover_ra_30m_utm43n_PT_5.tif')

        self.predicted_img = "./181126_predictedlandcover_ra_30m_utm43n_PT_5.tif"
        pixmap = QPixmap(self.predicted_img).scaledToWidth(520)
        pixmap.scaledToHeight(320)
        form.label_5.setPixmap(pixmap)
        form.label_5.resize(pixmap.width(), pixmap.height())

    # Метод получения подробной информации о работе приложения
    def get_help(self):
        info_window = InfoWindow("Основной функционал данной программы – семантическая сегментация спутниковых \n"
                                 "снимков и моделирование изменения земной поверхности на основе данных снимков.\n\n"
                                 "Для семантической сегментации снимка:\n"
                                 "1.	Загрузите изображение в формате .tiff, .png или .jpg, нажав на кнопку \n"
                                 "«загрузить» рядом с цифрой 1;\n"
                                 "2.	В случае успешной загрузки уменьшенная копия изображения появится рядом с \n"
                                 "кнопкой загрузки;\n"
                                 "3.	Запустите сегментацию изображения, нажав на кнопку «Сегментировать». \n"
                                 "В случае успеха сегментированное изображение появится на экране;\n"
                                 "4.	Сохраните полученное изображение, нажав на кнопку «Сохранить изображение».\n\n"
                                 "Для моделирования изменения земной поверхности:\n"
                                 "1.	Загрузите сегментированные изображения одной местности, сделанные за \n"
                                 "определённый промежуток времени в пункты списка, соответствующие \n"
                                 "порядковым номерам 1 и 2, ниже загрузите следующие дополнительные изображения:\n"
                                 "растровые карты расстояний до основных дорог, плотности населения \n"
                                 "и расстояний до крупных городов в формате .tiff, .png или .jpg;\n"
                                 "2.	Запустите моделирование изменения земной поверхности, нажав на кнопку \n"
                                 "«Спрогнозировать изменение». В случае успеха сегментированное \n"
                                 "изображение появится на экране;\n"
                                 "3.	Сохраните полученное изображение, нажав на кнопку «Сохранить изображение».",
                                 "О программе")
        info_window.exec_()


if __name__ == "__main__":
    app_controller = AppController()
    form.pushButton_2.clicked.connect(app_controller.start_segmentation)
    form.pushButton_3.clicked.connect(app_controller.start_prediction)
    form.pushButton_4.clicked.connect(app_controller.save_file)
    form.pushButton_5.clicked.connect(app_controller.get_help)
    form.pushButton_6.clicked.connect(app_controller.add_first_image)
    form.pushButton_7.clicked.connect(app_controller.remove_first_image)
    form.pushButton_9.clicked.connect(app_controller.add_second_image)
    form.pushButton_8.clicked.connect(app_controller.remove_second_image)
    form.pushButton_10.clicked.connect(app_controller.add_first_factor)
    form.pushButton_11.clicked.connect(app_controller.remove_first_factor)
    form.pushButton_12.clicked.connect(app_controller.add_second_factor)
    form.pushButton_14.clicked.connect(app_controller.remove_second_factor)
    form.pushButton_13.clicked.connect(app_controller.add_third_factor)
    form.pushButton_15.clicked.connect(app_controller.remove_third_factor)
    app.exec_()
