import numpy as np
from PIL import Image
import pandas as pd


# Метод формата вывода чисел с определённым количеством цифр после запятой
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


img = Image.open("./Land_Cover_Change/data/train/268881_mask.png")
img.resize((1024, 1024))
img2 = Image.open("./Land_Cover_Change/data/train/267065_mask.png")
img2.resize((1024, 1024))
# Преобразование изображений в матрицу RGB-значений пикселей
arr = np.array(img)
arr2 = np.array(img2)

"""

    urban_land - застроенная территория (строения)
    agriculture_land - сельскохозяйственные земли (пахотные земли)
    rangeland - пастбища (луг, саванна, степь, тундра и др.)
    forest_land - лес (широколиственные леса, смешанные леса, тайга)
    water - вода (реки, моря, океаны, озёра и др.)
    barren_land - пустошь (каменистая, песчаная, глинная почва)
    unknown - неизвестный тип (облака и др.)

"""
# Матрица переходов
matrix = [["", "urban_land", "agriculture_land", "rangeland", "forest_land", "water", "barren_land", "unknown", "sum"],
          ["urban_land", 0, 0, 0, 0, 0, 0, 0, 0], ["agriculture_land", 0, 0, 0, 0, 0, 0, 0, 0],
          ["rangeland", 0, 0, 0, 0, 0, 0, 0, 0],
          ["forest_land", 0, 0, 0, 0, 0, 0, 0, 0], ["water", 0, 0, 0, 0, 0, 0, 0, 0],
          ["barren_land", 0, 0, 0, 0, 0, 0, 0, 0],
          ["unknown", 0, 0, 0, 0, 0, 0, 0, 0]]

makeNewMatrix = True

# Заполнение матрицы переходов
if makeNewMatrix:
    for i in range(len(arr)):
        print(toFixed(i / 2447 * 100, 3), "%")
        for j in range(len(arr[i]) - 1):
            if (arr[i][j] == [0, 255, 255]).all() and (arr2[i][j] == [0, 255, 255]).all():  # urban_land to urban_land
                matrix[1][1] += 1
            elif (arr[i][j] == [0, 255, 255]).all() and (
                    arr2[i][j] == [255, 255, 0]).all():  # urban_land to agriculture_land
                matrix[1][2] += 1
            elif (arr[i][j] == [0, 255, 255]).all() and (arr2[i][j] == [255, 0, 255]).all():  # urban_land to rangeland
                matrix[1][3] += 1
            elif (arr[i][j] == [0, 255, 255]).all() and (arr2[i][j] == [0, 255, 0]).all():  # urban_land to forest_land
                matrix[1][4] += 1
            elif (arr[i][j] == [0, 255, 255]).all() and (arr2[i][j] == [0, 0, 255]).all():  # urban_land to water
                matrix[1][5] += 1
            elif (arr[i][j] == [0, 255, 255]).all() and (
                    arr2[i][j] == [255, 255, 255]).all():  # urban_land to barren_land
                matrix[1][6] += 1
            elif (arr[i][j] == [255, 255, 0]).all() and (
                    arr2[i][j] == [255, 255, 0]).all():  # agriculture_land to agriculture_land
                matrix[2][2] += 1
            elif (arr[i][j] == [255, 255, 0]).all() and (
                    arr2[i][j] == [0, 255, 255]).all():  # agriculture_land to urban_land
                matrix[2][1] += 1
            elif (arr[i][j] == [255, 255, 0]).all() and (
                    arr2[i][j] == [255, 0, 255]).all():  # agriculture_land to rangeland
                matrix[2][3] += 1
            elif (arr[i][j] == [255, 255, 0]).all() and (
                    arr2[i][j] == [0, 255, 0]).all():  # agriculture_land to forest_land
                matrix[2][4] += 1
            elif (arr[i][j] == [255, 255, 0]).all() and (arr2[i][j] == [0, 0, 255]).all():  # agriculture_land to water
                matrix[2][5] += 1
            elif (arr[i][j] == [255, 255, 0]).all() and (
                    arr2[i][j] == [255, 255, 255]).all():  # agriculture_land to barren_land
                matrix[2][6] += 1
            elif (arr[i][j] == [255, 0, 255]).all() and (arr2[i][j] == [0, 255, 255]).all():  # rangeland to urban_land
                matrix[3][1] += 1
            elif (arr[i][j] == [255, 0, 255]).all() and (
                    arr2[i][j] == [255, 255, 0]).all():  # rangeland to agriculture_land
                matrix[3][2] += 1
            elif (arr[i][j] == [255, 0, 255]).all() and (arr2[i][j] == [255, 0, 255]).all():  # rangeland to rangeland
                matrix[3][3] += 1
            elif (arr[i][j] == [255, 0, 255]).all() and (arr2[i][j] == [0, 255, 0]).all():  # rangeland to forest_land
                matrix[3][4] += 1
            elif (arr[i][j] == [255, 0, 255]).all() and (arr2[i][j] == [0, 0, 255]).all():  # rangeland to water
                matrix[3][5] += 1
            elif (arr[i][j] == [255, 0, 255]).all() and (
                    arr2[i][j] == [255, 255, 255]).all():  # rangeland to barren_land
                matrix[3][6] += 1
            elif (arr[i][j] == [0, 255, 0]).all() and (arr2[i][j] == [0, 255, 255]).all():  # forest_land to urban_land
                matrix[4][1] += 1
            elif (arr[i][j] == [0, 255, 0]).all() and (
                    arr2[i][j] == [255, 255, 0]).all():  # forest_land to agriculture_land
                matrix[4][2] += 1
            elif (arr[i][j] == [0, 255, 0]).all() and (arr2[i][j] == [255, 0, 255]).all():  # forest_land to rangeland
                matrix[4][3] += 1
            elif (arr[i][j] == [0, 255, 0]).all() and (arr2[i][j] == [0, 255, 0]).all():  # forest_land to forest_land
                matrix[4][4] += 1
            elif (arr[i][j] == [0, 255, 0]).all() and (arr2[i][j] == [0, 0, 255]).all():  # forest_land to water
                matrix[4][5] += 1
            elif (arr[i][j] == [0, 255, 0]).all() and (
                    arr2[i][j] == [255, 255, 255]).all():  # forest_land to barren_land
                matrix[4][6] += 1
            elif (arr[i][j] == [0, 0, 255]).all() and (arr2[i][j] == [0, 255, 255]).all():  # water to urban_land
                matrix[5][1] += 1
            elif (arr[i][j] == [0, 0, 255]).all() and (arr2[i][j] == [255, 255, 0]).all():  # water to agriculture_land
                matrix[5][2] += 1
            elif (arr[i][j] == [0, 0, 255]).all() and (arr2[i][j] == [255, 0, 255]).all():  # water to rangeland
                matrix[5][3] += 1
            elif (arr[i][j] == [0, 0, 255]).all() and (arr2[i][j] == [0, 255, 0]).all():  # water to forest_land
                matrix[5][4] += 1
            elif (arr[i][j] == [0, 0, 255]).all() and (arr2[i][j] == [0, 0, 255]).all():  # water to water
                matrix[5][5] += 1
            elif (arr[i][j] == [0, 0, 255]).all() and (arr2[i][j] == [255, 255, 255]).all():  # water to barren_land
                matrix[5][6] += 1
            elif (arr[i][j] == [255, 255, 255]).all() and (
                    arr2[i][j] == [0, 255, 255]).all():  # barren_land to urban_land
                matrix[6][1] += 1
            elif (arr[i][j] == [255, 255, 255]).all() and (
                    arr2[i][j] == [255, 255, 0]).all():  # barren_land to agriculture_land
                matrix[6][2] += 1
            elif (arr[i][j] == [255, 255, 255]).all() and (
                    arr2[i][j] == [255, 0, 255]).all():  # barren_land to rangeland
                matrix[6][3] += 1
            elif (arr[i][j] == [255, 255, 255]).all() and (
                    arr2[i][j] == [0, 255, 0]).all():  # barren_land to forest_land
                matrix[6][4] += 1
            elif (arr[i][j] == [255, 255, 255]).all() and (arr2[i][j] == [0, 0, 255]).all():  # barren_land to water
                matrix[6][5] += 1
            elif (arr[i][j] == [255, 255, 255]).all() and (
                    arr2[i][j] == [255, 255, 255]).all():  # barren_land to barren_land
                matrix[6][6] += 1

    for i in range(1, 8):
        sum = 0
        for j in range(1, 8):
            sum += matrix[i][j]
        matrix[i][8] = sum

    frame1 = pd.DataFrame(matrix)
    # Сохранение матрицы переходов
    frame1.to_csv('./frame.csv', sep='\t', encoding='utf-8')

    # Создание матрицы вероятностей переходов
    matrix2 = matrix

    # Заполнение матрицы вероятностей переходов
    for i in range(1, 8):
        for j in range(1, 8):
            if (matrix2[i][8] != 0) and (matrix2[i][j] != 0):
                matrix2[i][j] = toFixed(matrix2[i][j] / matrix2[i][8], 5)

    frame2 = pd.DataFrame(matrix2)
    # Сохранение матрицы вероятностей переходов
    frame2.to_csv('./frame2.csv', sep='\t', encoding='utf-8')
