import argparse
from lightning_wrapper import LightningSegmentationModule, CSVLoggerMetricsAndConfusion

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from torchmetrics import classification
from torchmetrics import segmentation

import os
import pandas as pd
import numpy as np
import json
import yaml

from typing import Dict, List

import torch
from torch import nn

from models.models_factories import (
    criterion_factory_dict,
    optimizers_factory_dict,
    lr_schedulers_factory_dict,
    segmentation_nns_factory_dict,
    create_model,
    create_augmentation_transforms)
from data import SegmentationDataset

from datetime import datetime
from itertools import combinations

def create_and_train_moodel(config_dict: Dict, path_to_saving_dir: str):
    path_to_dataset_root = config_dict['path_to_dataset_root']

    path_to_dataset_info_csv = os.path.join(path_to_dataset_root, 'data_info_table.csv')
    path_to_surface_classes_json = os.path.join(path_to_dataset_root, 'surface_classes.json')

    input_image_size = config_dict['input_image_size']
    multispecter_bands_indices = config_dict['multispecter_bands_indices']
    device = config_dict['device']

    # чтение списка имен классов поверхностей
    with open(path_to_surface_classes_json) as fd:
        surface_classes_list = json.load(fd)
    # чтение таблицы с информацией о каждом изображении в выборке
    images_df = pd.read_csv(path_to_dataset_info_csv)

    path_to_partition_json = os.path.join(path_to_dataset_root, 'dataset_partition.json')
    # чтение словаря со списками квадратов, находящихся в обучающей и тестовой выборке
    with open(path_to_partition_json) as fd:
        partition_dict = json.load(fd)

    # формирование pandas DataFrame-ов с информацией об изображениях обучающей и тестовой выборках
    train_images_df = []
    for train_square in partition_dict['train_squares']:
        train_images_df.append(images_df[images_df['square_id']==train_square])
    train_images_df = pd.concat(train_images_df, ignore_index=True)

    test_images_df = []
    for test_square in partition_dict['test_squares']:
        test_images_df.append(images_df[images_df['square_id']==test_square])
    test_images_df = pd.concat(test_images_df, ignore_index=True)

    #train_images_df, test_images_df = train_test_split(images_df, test_size=0.3, random_state=0)

    class_num = images_df['class_num'].iloc[0]

    # формирование словаря, отображающейго имя класса поверхности в индекс класса
    class_name2idx_dict = {n:i for i, n in enumerate(surface_classes_list)}

    # вычисление распределений пикселей в классах поверхностей 
    classes_pixels_distribution_df = images_df[surface_classes_list]
    classes_pixels_num = classes_pixels_distribution_df.sum()
    classes_weights = classes_pixels_num / classes_pixels_num.sum()
    classes_weights = classes_weights[surface_classes_list].to_numpy().astype(np.float32)


    '''
    train_transforms = v2.Compose(
        [v2.Resize((input_image_size,input_image_size), antialias=True),v2.ToDtype(torch.float32, scale=True)])
    test_transforms = v2.Compose(
        [v2.Resize((input_image_size,input_image_size), antialias=True),v2.ToDtype(torch.float32, scale=True)])
    '''
    train_transforms = create_augmentation_transforms(config_dict['train_augmentations']) 
    test_transforms = nn.Identity()
    # если ф-ция потерь перекрестная энтропия, то проверяем, есть ли там веса классов
    if config_dict['loss']['type'] == 'crossentropy':
        # если в параметрах функции потерь стоит строка 'classes', надо передать в функцию вектор весов классов
        if 'weight' in config_dict['loss']['params']:
            if isinstance(config_dict['loss']['params']['weight'], (list, tuple)):
                config_dict['loss']['params']['weight'] = torch.tensor(config_dict['loss']['params']['weight'])
            
            elif config_dict['loss']['params']['weight'] is not None:
                config_dict['loss']['params']['weight'] = torch.tensor(classes_weights)

    # создание функции потерь
    criterion = criterion_factory_dict[config_dict['loss']['type']](**config_dict['loss']['params'])

    # если ф-ция потерь перекрестная энтропия, то проверяем, есть ли там веса классов
    if config_dict['loss']['type'] == 'crossentropy':
        # если в параметрах функции потерь стоит строка 'classes', надо передать в функцию вектор весов классов
        if 'weight' in config_dict['loss']['params']:
            if isinstance(config_dict['loss']['params']['weight'], torch.Tensor):
                config_dict['loss']['params']['weight'] = config_dict['loss']['params']['weight'].cpu().tolist()

    model = create_model(config_dict, segmentation_nns_factory_dict)
    model = model.to(device)

    # создаем датасеты и даталоадеры
    train_dataset = SegmentationDataset(path_to_dataset_root=path_to_dataset_root, samples_df=train_images_df, channel_indices=multispecter_bands_indices, transforms=train_transforms, dtype=torch.float32, device=device)
    test_dataset = SegmentationDataset(path_to_dataset_root=path_to_dataset_root, samples_df=test_images_df, channel_indices=multispecter_bands_indices, transforms=test_transforms, dtype=torch.float32, device=device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config_dict['batch_size'])

    # тестовое чтение данных
    for data, labels in test_loader:
        break

    # тестовая обработка данных нейронной сетью
    ret = model(data)
    print('Test model inference:')
    print(f'input_shape:{data.shape}, output_shape:{ret.shape}')

    createion_time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    nn_arch_str = config_dict["segmentation_nn"]["nn_architecture"]
    nn_encoder_str = config_dict["segmentation_nn"]["params"]["encoder_name"]
    name_postfix = config_dict["name_postfix"]
    if name_postfix is not None:
        model_name = f'{nn_arch_str}_{nn_encoder_str}_{name_postfix} {createion_time_str}'
    else:
        model_name = f'{nn_arch_str}_{nn_encoder_str} {createion_time_str}'

    epoch_num = config_dict['epoch_num']

    print('----------------------------------------------------------')
    print('Created model:')
    print(f'{model_name}')
    print('----------------------------------------------------------')
    print()

    # создаем список словарей с информацией о вычисляемых метриках с помощью multiclass confusion matrix
    # см. подробнее ддокументацию к функции compute_metric_from_confusion
    metrics_dict = {
        'train': {
            'iou': classification.JaccardIndex(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'precision': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'recall': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'confusion': classification.ConfusionMatrix(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
        },
        'val': {
            'iou': classification.JaccardIndex(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'precision': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'recall': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'confusion': classification.ConfusionMatrix(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
        }
    }

    optimizer_cfg = {
        'optmizer': optimizers_factory_dict[config_dict['optimizer']['type']],
        'optimizer_args':config_dict['optimizer']['args'],
        'lr_scheduler': lr_schedulers_factory_dict[config_dict['lr_scheduler']['type']],
        'lr_scheduler_args': config_dict['lr_scheduler']['args'],
        'lr_scheduler_params': config_dict['lr_scheduler']['params']
    }

    # Создаем модуль Lightning
    segmentation_module = LightningSegmentationModule(model, criterion, optimizer_cfg, metrics_dict, class_name2idx_dict)

    # задаем путь до папки с логгерами и создаем логгер, записывающий результаты в csv
    #path_to_saving_dir = 'saving_dir'
    csv_logger = CSVLoggerMetricsAndConfusion(
        save_dir = path_to_saving_dir,
        name=model_name, 
        flush_logs_every_n_steps=1,
        )

    # создаем объект, записывающий в чекпоинт лучшую модель
    path_to_save_model_dir = os.path.join(path_to_saving_dir, model_name)
    os.makedirs(path_to_save_model_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        mode="max",
        filename=model_name+"-{epoch:02d}-{val_iou_mean:.3}",
        dirpath=path_to_save_model_dir, 
        save_top_k=1, monitor="val_iou_mean"
        )

    trainer = L.Trainer(logger=[csv_logger],
            max_epochs=epoch_num, 
            callbacks=[checkpoint_callback],
            accelerator = 'gpu'
            )

    # сохраняем конфигурацию
    path_to_config = os.path.join(path_to_save_model_dir, 'training_config.yaml')
    with open(path_to_config, 'w', encoding='utf-8') as fd:
        #json.dump(config_dict, fd, indent=4)
        yaml.dump(config_dict, fd, indent=4)

    trainer.fit(segmentation_module , train_loader, test_loader)

    print()
    print(('----------------------------------------------------------'))
    print(f'Training {model_name} is over for {epoch_num} epochs')
    print(('----------------------------------------------------------'))
    print()

def search_best_multispecter_bands_combination(config_dict: Dict, path_to_saving_dir:str, basic_bands_indices: List):
    '''
    Выполняется перебор всех возможных комбинаций 
    '''
    experiment_date = datetime.now().strftime('%Y-%m%dT%H-%M-%S')
    path_to_experiment_saving_dir = os.path.join(path_to_saving_dir, 'best_bands_search', f'experiment_{experiment_date}')
    os.makedirs(path_to_experiment_saving_dir, exist_ok=True)
    multispecter_bands_indices = config_dict['multispecter_bands_indices']
    rest_bands_indices = set(multispecter_bands_indices) - set(basic_bands_indices)
    rest_bands_indices = list(rest_bands_indices)
    rest_bands_indices = [x for x in rest_bands_indices if isinstance(x, int)] + [x for x in rest_bands_indices if isinstance(x, str)]
    rest_bands_num = len(rest_bands_indices)
    combinations_num = 2**rest_bands_num
    combination_cnt = 1
    # проверяем все возможные комбинации оставшихся каналов
    for k in range(rest_bands_num):
        #k+=1
        for combination_of_indices in combinations(rest_bands_indices, k):
            bands_indices_to_test = basic_bands_indices + list(combination_of_indices)
            bands_indices_to_test = sorted([x for x in bands_indices_to_test if isinstance(x, int)]) + sorted([x for x in bands_indices_to_test if isinstance(x, str)])
            config_dict['multispecter_bands_indices'] = bands_indices_to_test
            print('############################################')
            print(f'# Train combination {combination_of_indices} #{combination_cnt} of total {combinations_num}')
            print('############################################')
            create_and_train_moodel(config_dict, path_to_experiment_saving_dir)
            combination_cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_config', nargs='+')
    parser.add_argument('--training_mode', help='Mode of training. Available options: "single_nn", "search_best_bands"')
    parser.add_argument('--path_to_saving_dir')

    sample_args = [
        '--path_to_config', 'training_configs/att_unet_efficientnet-b0_win_cat_agg.yaml',
        '--training_mode', 'train_nns',
        '--path_to_saving_dir', 'saving_dir'
    ]
    args = parser.parse_args(sample_args)
    paths_to_configs = args.path_to_config
    training_mode = args.training_mode
    path_to_saving_dir = args.path_to_saving_dir

    
    if training_mode == 'train_nns':
        for path_to_config in paths_to_configs:
            with open(path_to_config) as fd:
                if path_to_config.endswith('.yaml'):
                    config_dict = yaml.load(fd, Loader=yaml.Loader)
                elif path_to_config.endswith('.json'):
                    config_dict = json.load(fd)
            create_and_train_moodel(config_dict, path_to_saving_dir)
    elif training_mode == 'search_best_bands':
        path_to_config = paths_to_configs[0]
        # чтение файла конфигурации
        with open(path_to_config) as fd:
            if path_to_config.endswith('.yaml'):
                config_dict = yaml.load(fd, Loader=yaml.Loader)
            elif path_to_config.endswith('.json'):
                config_dict = json.load(fd)
        search_best_multispecter_bands_combination(config_dict, path_to_saving_dir, basic_bands_indices=[1, 2, 3, 7])


    

