import os, cv2

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


# Демонстрация полученных изображений и масок
def visualize_data(**images):
    plt.figure(figsize=(20, 8))
    for index, (name, image) in enumerate(images.items()):
        plt.subplot(1, len(images), index + 1)
        plt.xticks([]);
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


# Преобразование массива меток сегментированного изображения в формат one-hot
# путем замены каждого значения пикселя вектором длины num_classes
def one_hot_encode(image, image_values):
    """
    # Arguments
        label: Сегментированное изображение в формате массива
    # Returns
        Двумерный массив с той же шириной и высотой, что и входные данные, но
        с размером глубины num_classes
    """
    semantic_map = []
    for color in image_values:
        type_map = np.all(np.equal(image, color), axis=-1)
        semantic_map.append(type_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    """
    Преобразование двумерного массива в одноканальном формате (глубина - num_classes)
    в двумерный массив такого же размера только с 1 каналом, где каждое значение пикселя равно
    номеру типа земного покрова.

    """
    return np.argmax(image, axis=-1)


# Цветовая кодировка загруженного изображения
def color_segmentation(image, label_values):
    """
    # Arguments
        image: одноканальный массив, где каждое значение пикселя равно
        номеру типа земного покрова.

    # Returns
       Изображение с цветовой кодировкой для дальнейшей визуализации

    """
    colour_codes = np.array(label_values)

    return colour_codes[image.astype(int)]


# Масштабирование изображение до заданного размера
def get_training_transformation():
    train_transformation = [
        transformation.RandomCrop(height=1024, width=1024, always_apply=True),
        transformation.HorizontalFlip(p=0.5),
        transformation.VerticalFlip(p=0.5),
    ]
    return transformation.Compose(train_transformation)


# Обрезает изображение до заданного размера
def get_validation_transformation():
    train_transform = [
        transformation.CenterCrop(height=1024, width=1024, always_apply=True),
    ]
    return transformation.Compose(train_transform)


# Транспонирование тензора
def to_tensor(x, **kwargs):
    """

    OpenCV загружает изображение с HWC-макетом (высота, ширина, каналы),
    в то время как Pytorch требует CHW-layout (каналы, высота, ширина)

    """
    return x.transpose(2, 0, 1).astype('float32')


# Предварительная обработка изображения
def get_preprocessing(preprocessing_function=None):
    """
    Аргументы:
    preprocessing_fn: функция нормализации обработки данных индивидуально для каждой обученной нейронной сети
    """
    transformations = []
    if preprocessing_function:
        transformations.append(transformation.Lambda(image=preprocessing_function))
    transformations.append(transformation.Lambda(image=to_tensor, mask=to_tensor))

    return transformation.Compose(transformations)


# Класс, представляющий утилиты – вспомогательные методы для проведения семантической сегментации изображения
class LandCoverChangeDataset(torch.utils.data.Dataset):
    """
    Считывание и преобразование изображений (увеличение и предварительная обработка)

        Args:
            data_frame (string): данные о путях к изображениям и маскам
            type_color (list): RGB значения выбранных типов земной поверхности
            transformation (albumentations.Compose): общие преобразования изображений (масштабирование, поворот и пр.)
            preprocessing (albumentations.Compose): предварительная обработка данных (стандартизация, удаление среднего значения и масштабирование дисперсии и др.)
    """

    # Конструктор класса
    def __init__(
            self,
            data_frame,
            type_color=None,
            transformation=None,
            preprocessing=None,
    ):
        # Поле, содержащее пути до изображений для обучения нейронной сети
        self.image_paths = data_frame['sat_image_path'].tolist()
        # Поле, содержащее пути до масок для изображений для обучения нейронной сети
        self.mask_paths = data_frame['mask_path'].tolist()

        # Поле, содержащее соответствие цветовых RGB-значений для заданных типов поверхностей
        self.type_color = type_color
        # Поле для обработанного и отмасштабированного изображения
        self.transformation = transformation
        self.preprocessing = preprocessing

    # Метод получения и обработки по заданных параметрам изображений и масок, загруженных в программу для
    # семантической сегментации
    def __getitem__(self, index):
        # Получение информации об изображениях и масках
        image = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[index]), cv2.COLOR_BGR2RGB)

        # Создание конвертированной маски
        mask = one_hot_encode(mask, self.type_color).astype('float')

        # Выполнение заданных преобразований
        if self.transformation:
            sample = self.transformation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Выполнение заданной предварительной обработки
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    # Метод получения данных о количестве загруженных данных для обучения нейронной сети
    def __len__(self):
        return len(self.image_paths)


data_folder = 'data/'

# Получение данных для обучения и тестирования
data = pd.read_csv(os.path.join(data_folder, 'metadata.csv'))
data = data[data['split'] == 'train']
data = data[['image_id', 'sat_image_path', 'mask_path']]
data['sat_image_path'] = data['sat_image_path'].apply(lambda image_path: os.path.join(data_folder, image_path))
data['mask_path'] = data['mask_path'].apply(lambda image_path: os.path.join(data_folder, image_path))

"""
    Аргумент ключевого слова frac определяет долю строк, возвращаемых в случайной выборке,
    поэтому frac = 0.3 означает возврат 30% от общего числа строк (в случайном порядке)
"""
data = data.sample(frac=0.3).reset_index(drop=True)

# Раздедение данных для обучения и валидации
valid_data = data.sample(frac=0.1, random_state=42)
train_data = data.drop(valid_data.index)
print(len(train_data), len(valid_data))

# Получение данных (название и цветое rgb значение) о всех возможных типах земной поверхности для выполнения
# семантической сегментации
land_types = pd.read_csv(os.path.join(data_folder, 'class_dict.csv'))
land_types_names = land_types['name'].tolist()
land_types_colors = land_types[['r', 'g', 'b']].values.tolist()

print('Все типы земной поверхности и соответствующие им цветовые значения RGB:')
print('Тип земной поверхности: ', land_types_names)
print('Цветовое значение: ', land_types_colors)

"""

    urban_land - застроенная территория (строения)
    agriculture_land - сельскохозяйственные земли (пахотные земли)
    rangeland - пастбища (луг, саванна, степь, тундра и др.)
    forest_land - лес (широколиственные леса, смешанные леса, тайга)
    water - вода (реки, моря, океаны, озёра и др.)
    barren_land - пустошь (каменистая, песчаная, глинная почва)
    unknown - неизвестный тип (облака и др.)
    
"""
# Необходимые имена типов земной поверхности
required_types = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']

# Получение цветовых rgb значений для необходимых типов земной поверхности
required_types_indices = [land_types_names.index(land_type.lower()) for land_type in required_types]
required_types_colors = np.array(land_types_colors)[required_types_indices]

dataset = LandCoverChangeDataset(train_data, type_color=required_types_colors)
image, mask = dataset[2]

augmented_dataset = LandCoverChangeDataset(
    train_data,
    transformation=get_training_transformation(),
    type_color=required_types_colors,
)

# Демонстрация трёх примеров изображения и маски с семантической сегментацией земного покрова
for index in range(3):
    image, mask = augmented_dataset[index]
    visualize_data(
        original_image=image,
        ground_truth_mask=color_segmentation(reverse_one_hot(mask), required_types_colors)
    )

# Создание модели сегментации с предварительно обученным енкодером
model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=len(required_types),
    activation='sigmoid',
)

preprocessing_data = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')

# Получение экземпляров набора данных
train_dataset = LandCoverChangeDataset(
    train_data,
    transformation=get_training_transformation(),
    preprocessing=get_preprocessing(preprocessing_data),
    type_color=required_types_colors,
)

valid_dataset = LandCoverChangeDataset(
    valid_data,
    transformation=get_validation_transformation(),
    preprocessing=get_preprocessing(preprocessing_data),
    type_color=required_types_colors,
)

"""
    batch_size - количество выборок, которые будут распространяться по нейронной сети
    num_workers - многопроцессорную загрузку данных с указанным количеством рабочих процессов загрузчика
"""
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# Флаг, показывающий, нужно ли выполнять тренировку нейронной сети
TRAINING = True

# Количество эпох: 1 эпоха = один проход вперед и один проход назад по всем примерам обучения
epochs = 5

"""
   torch.cuda добавляет поддержку типов тензоров CUDA, 
   которые реализуют ту же функцию, что и тензоры процессора,
   но используют графические процессоры для вычислений
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определение функции потерь — функция, которая характеризует потери
# при неправильном принятии решений на основе наблюдаемых данных
loss = smp.utils.losses.DiceLoss()

# Определение показателей
# IoU - это метрика для количественной оценки точности алгоритма сегментации
# путем перекрытия предсказаний, сделанных моделью, с основной истиной.
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

"""
    torch.optim - пакет, реализующий различные алгоритмы оптимизации.
    Adam - ​​алгоритм оптимизации, который можно использовать вместо классической процедуры
    стохастического градиентного спуска для итеративного обновления весов сети на основе обучающих данных.
    lr (float, optional) – learning rate (default: 1e-3)
"""
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.00008),
])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)

# Обучение сети
if TRAINING:

    best_result = 0.0
    train_logs_list, valid_logs_list = [], []
    os.environ["OMP_NUM_THREADS"] = "1"
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        os.environ["OMP_NUM_THREADS"] = "1"
        # Сохранение модели с наилучшим результатом обучения
        if best_result < valid_logs['iou_score']:
            best_result = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

# Загрузка наилучшей сохраненной модели из текущего запуска программы
if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location=device)
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

best_model = model

for index in range(len(test_dataset)):
    image, gt_mask = test_dataset[index]
    image_visualization = test_dataset_visualization[index][0].astype('uint8')
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
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

# Оценка модели на тестовом наборе данных
test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)

valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

# Построение графика потерь и IoU метрики для тестовых надобор и наборов для валидации
train_logs_dataframe = pd.DataFrame(train_logs_list)
valid_logs_dataframe = pd.DataFrame(valid_logs_list)
train_logs_dataframe.T

plt.figure(figsize=(20, 8))
plt.plot(train_logs_dataframe.index.tolist(), train_logs_dataframe.iou_score.tolist(), lw=3, label='Train')
plt.plot(valid_logs_dataframe.index.tolist(), valid_logs_dataframe.iou_score.tolist(), lw=3, label='Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('IoU Score', fontsize=20)
plt.title('IoU Score Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('iou_score_plot.png')
plt.show()

plt.figure(figsize=(20, 8))
plt.plot(train_logs_dataframe.index.tolist(), train_logs_dataframe.dice_loss.tolist(), lw=3, label='Train')
plt.plot(valid_logs_dataframe.index.tolist(), valid_logs_dataframe.dice_loss.tolist(), lw=3, label='Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Dice Loss', fontsize=20)
plt.title('Dice Loss Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('dice_loss_plot.png')
plt.show()
