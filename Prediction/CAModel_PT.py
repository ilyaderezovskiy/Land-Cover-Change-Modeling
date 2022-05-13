"""
Following is the order of the land cover data:
    Built-up land --> Class 1
    Vegetation --> Class 2
    Water body --> Class 3
    Others --> Class 4
"""

import os, math
import numpy as np
import gdal
from copy import deepcopy


# Чтение растрового изображения и представление его в виде массива
def read_raster_image(file):
    image = gdal.Open(file)
    arr = image.GetRasterBand(1)
    return image, arr.ReadAsArray()


def identicalList(input_list):
    global logical
    input_list = np.array(input_list)
    logical = input_list == input_list[0]
    if sum(logical) == len(input_list):
        return True
    else:
        return False


def builtupAreaDifference(landcover1, landcover2, buclass=1, cellsize=30):
    return (sum(sum(((landcover2 == buclass).astype(int) - (landcover1 == buclass).astype(int)) != 0)) * (
            cellsize ** 2) / 1000000)


# Класс, содержащий обработанные и подготовленные для моделирования изменения земной поверхности изображения
class LandCover():

    # Конструктор класса
    def __init__(self, file_1, file_2):
        # Поле загруженного изображения для моделирования изменения земной поверхности
        # Поле загруженного изображения для моделирования изменения земной поверхности в виде мтарицы
        self.image_1, self.arr_image_1 = read_raster_image(file_1)
        # Поле загруженного изображения для моделирования изменения земной поверхности
        # Поле загруженного изображения для моделирования изменения земной поверхности в виде мтарицы
        self.image_2, self.arr_image_2 = read_raster_image(file_2)
        self.сheck_correctness()

    # Проверка корректности и характеристик изображений
    def сheck_correctness(self):
        # Проверка размеров изображений
        if (self.image_1.RasterXSize == self.image_2.RasterXSize) and (
                self.image_1.RasterYSize == self.image_2.RasterYSize):
            # Поле количества строк полученной матрицы из изображения
            # Поле количества столбцов полученной матрицы из изображения
            self.rows, self.columns = (self.image_1.RasterYSize, self.image_1.RasterXSize)
        else:
            print("Загруженные изображения имеют разный размер!")
        # Проверка соответствия типов земной поверхности на изображениях
        if (self.arr_image_1.max() == self.arr_image_2.max()) and (self.arr_image_1.min() == self.arr_image_2.min()):
            self.nFeature = (self.arr_image_1.max() - self.arr_image_1.min())
        else:
            print("Загруженные изображения имеют разное количество типов земной поверхности / размер!")

    # Создание нормализованной матрицы переходов
    def transition_matrix(self):
        self.matrix = np.random.randint(1, size=(self.nFeature, self.nFeature))
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                pixel_1 = self.arr_image_1[x, y]
                pixel_2 = self.arr_image_2[x, y]
                self.matrix[pixel_1 - 1, pixel_2 - 1] += 1
        # Поле, содержащее нормализованную матрицу переходов
        self.matrix_norm = np.random.randint(1, size=(4, 5)).astype(float)
        # Создание нормализованной матрицы переходов
        for x in range(0, self.matrix.shape[0]):
            for y in range(0, self.matrix.shape[1]):
                self.matrix_norm[x, y] = self.matrix[x, y] / (self.matrix[x, :]).sum()


# Класс, содержащий обработанные и подготовленные и факторов для модели клеточного автомата
class Factors():

    # Конструктор класса
    def __init__(self, *files):
        self.factors = dict()
        self.factors_ds = dict()
        self.count_factors = len(files)
        index = 1
        for file in files:
            self.factors_ds[index], self.factors[index] = read_raster_image(file)
            index += 1
        self.сheck_correctness()

    # Метод проверки загруженных факторов
    def сheck_correctness(self):
        rows = []
        columns = []
        for i in range(1, self.count_factors + 1):
            rows.append(self.factors_ds[i].RasterYSize)
            columns.append(self.factors_ds[i].RasterXSize)
        if (identicalList(rows) == True) and ((identicalList(columns) == True)):
            self.rows = self.factors_ds[i].RasterYSize
            self.columns = self.factors_ds[i].RasterXSize
        else:
            print("Введённые изображения факторов имеют различные размеры.")


# Класс клеточного автомата для моделирования изменения земной поверхности
class CellularAutomatonModel():

    # Конструктор класса
    def __init__(self, land_cover, factors):
        # Поле загруженного изображения для моделирования изменения земной поверхности
        self.land_covers = land_cover
        # Поле загруженных факторов для моделирования изменения земной поверхности
        self.factors = factors
        self.сheck_correctness()
        # Поле размера ядра
        self.kernel_size = 3

    # Метод проверки размеров загруженных изображений и факторов
    def сheck_correctness(self):
        if (self.land_covers.row == self.factors.row) and (self.factors.col == self.factors.col):
            self.rows = self.factors.row
            self.columns = self.factors.col
        else:
            print("Размер спутникового изображения и изображений-факторов должны совпадать!")

    # Метод установки пороговых значений
    def set_threshold(self, builtup_threshold, *other_thresholds_in_sequence):
        # Поле пороговых значений
        self.threshold = list(other_thresholds_in_sequence)
        self.builtupThreshold = builtup_threshold
        if len(self.threshold) == (len(self.factors.gf)):
            print("\nПороговые значения успешно установлены!")
        else:
            print('Неверное количество изображений-факторов!')

    # Метод моделирования изменения земной поверхности
    def predict(self):
        # Поле матрицы смоделированного изображения
        self.predicted = deepcopy(self.land_covers.arr_lc1)
        sideMargin = math.ceil(self.kernel_size / 2)
        for y in range(sideMargin, self.rows - (sideMargin - 1)):
            for x in range(sideMargin, self.columns - (sideMargin - 1)):
                kernel = self.land_covers.arr_lc1[y - (sideMargin - 1):y + (sideMargin),
                         x - (sideMargin - 1):x + (sideMargin)]
                builtupCount = sum(sum(kernel == 1))
                # Если количество заполненных ячеек больше или равно заданному пороговому значению
                if (builtupCount >= self.builtupThreshold) and (
                        self.factors.gf[5][y, x] != 1):
                    for factor in range(1, self.factors.nFactors + 1):
                        if self.threshold[factor - 1] < 0:
                            if (self.factors.gf[factor][y, x] <= abs(self.threshold[factor - 1])):
                                self.predicted[y, x] = 1
                            else:
                                pass
                        elif self.threshold[factor - 1] > 0:
                            if (self.factors.gf[factor][y, x] >= self.threshold[factor - 1]):
                                self.predicted[y, x] = 1
                            else:
                                pass
                if (y % 500 == 0) and (x % 500 == 0):
                    print("Row: %d, Col: %d, Builtup cells count: %d\n" % (y, x, builtupCount), end="\r", flush=True)

    # Метод сохранения полученного изображения
    def save_result(self, ouput_file_name):
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(ouput_file_name, self.columns, self.rows, 1, gdal.GDT_UInt16)
        outdata.SetGeoTransform(self.land_covers.ds_lc1.GetGeoTransform())
        outdata.SetProjection(self.land_covers.ds_lc1.GetProjection())
        outdata.GetRasterBand(1).WriteArray(self.predicted)
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()
        outdata = None


# Каталог, в котором находятся файлы
os.chdir("F:\\Cities_CA_Delhi\\ModelDevelopment")

# Изображения по которым будет моделироваться изменение земной поверхности
image_1 = "Actual_1994_sat.tif"
image_2 = "Actual_1999_sat.tif"

# Input all the parameters
cbd = "city_center_distance_factor.tif"
road = "road_distance_factor.tif"
restricted = "dda_2021_government_restricted.tif"
pop91 = "population_density_factor_1991.tif"
pop01 = "population_density_factor_2001.tif"
pop11 = "population_density_factor_2011.tif"
slope = "slope.tif"

land_cover = LandCover(image_1, image_2)

factors = Factors(cbd, road, pop01, slope, restricted)

cellular_automaton = CellularAutomatonModel(land_cover, factors)

# Set the threshold values, Assign negative threshold values if less than rule is required
# Based on the statistical and spatial accuracy displayed, the thresholds should be tweaked
cellular_automaton.set_threshold(3, -15000, -10000, 8000, -3, -1)

# Запуск моделирования изменения земной поверхности
cellular_automaton.predict()

# Сохранение полученного изображения
cellular_automaton.save_result('181126_predictedlandcover_ra_30m_utm43n_PT_5.tif')
