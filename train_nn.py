import argparse
from lightning_wrapper import LightningSegmentationModule, CSVLoggerMetricsAndConfusion

import lightning as L

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from torchmetrics import classification, nominal
from torchmetrics import segmentation

from copy import deepcopy

import random

import os
import pandas as pd
import numpy as np
import json
import yaml
import time
from datetime import timedelta

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
from data import SegmentationDataset, HSI_dataset

from datetime import datetime
from itertools import combinations, product

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

def create_seismic_sensors_dataset(config_dict):

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

    if 'crossval_iteration' in config_dict:
        partition_name = f'dataset_partition_{config_dict["crossval_iteration"]}.json'
    else:
        partition_name = 'dataset_partition.json'
    path_to_partition_json = os.path.join(path_to_dataset_root, partition_name)
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

        # создаем датасеты и даталоадеры
    train_dataset = SegmentationDataset(path_to_dataset_root=path_to_dataset_root, samples_df=train_images_df, channel_indices=multispecter_bands_indices, transforms=train_transforms, dtype=torch.float32, device=device)
    test_dataset = SegmentationDataset(path_to_dataset_root=path_to_dataset_root, samples_df=test_images_df, channel_indices=multispecter_bands_indices, transforms=test_transforms, dtype=torch.float32, device=device)
    generator_params = {}
    if 'deterministic_seed' in config_dict:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        #g = torch.Generator(device=config_dict['device'])
        g = torch.Generator()
        g.manual_seed(config_dict['deterministic_seed']) 
        generator_params['generator'] = g
        generator_params['worker_init_fn'] = seed_worker

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True, **generator_params)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config_dict['batch_size'])

    return train_loader, test_loader, class_name2idx_dict, classes_weights

def create_hsi_uav_dataset(config_dict):
    path_to_dataset_root = config_dict['path_to_dataset_root']
    device = config_dict['device']
    input_image_size = config_dict['input_image_size']

    train_transforms = create_augmentation_transforms(config_dict['train_augmentations']) 
    test_transforms = nn.Identity()

    train_dataset = HSI_dataset(
        path_to_dataset_partition=os.path.join(path_to_dataset_root, 'Train', 'Training'),
        augmentation_transforms=train_transforms,
        device=device
        )

    val_dataset = HSI_dataset(
        path_to_dataset_partition=os.path.join(path_to_dataset_root, 'Train', 'Validation'),
        augmentation_transforms=test_transforms,
        device=device
        )

    test_dataset = HSI_dataset(
        path_to_dataset_partition=os.path.join(path_to_dataset_root, 'Test'),
        augmentation_transforms=test_transforms,
        device=device
        )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config_dict['batch_size'])
    class_name2idx_dict = {f'{i}':i  for i in range(30)}
    classes_weights = np.array([1. for c in class_name2idx_dict]).astype(np.float32)
    return train_loader, test_loader, class_name2idx_dict, classes_weights
   
def create_seismic_metrics(class_name2idx_dict, device):
    metrics_dict = {
        'train': {
            'iou': classification.JaccardIndex(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'precision': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'recall': classification.Recall(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'confusion': classification.ConfusionMatrix(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
        },
        'val': {
            'iou': classification.JaccardIndex(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'precision': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'recall': classification.Recall(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'confusion': classification.ConfusionMatrix(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
        }
    }
    return metrics_dict

def create_hsi_uav_metrics(class_name2idx_dict, device):
    metrics_dict = {
        'train': {
            'iou': classification.JaccardIndex(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'micro_avg_iou': classification.JaccardIndex(task='multiclass', average='micro', num_classes=len(class_name2idx_dict)).to(device),
            'precision': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'micro_avg_precision': classification.Precision(task='multiclass', average='micro', num_classes=len(class_name2idx_dict)).to(device),
            'recall': classification.Recall(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'micro_avg_recall': classification.Recall(task='multiclass', average='micro', num_classes=len(class_name2idx_dict)).to(device),
            'accuracy':  classification.Accuracy(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
            'macro_avg_acc':  classification.Accuracy(task='multiclass', average='macro', num_classes=len(class_name2idx_dict)).to(device),
            'kappa': classification.CohenKappa(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
            'confusion': classification.ConfusionMatrix(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
        },
        'val': {
            'iou': classification.JaccardIndex(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'micro_avg_iou': classification.JaccardIndex(task='multiclass', average='micro', num_classes=len(class_name2idx_dict)).to(device),
            'precision': classification.Precision(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'micro_avg_precision': classification.Precision(task='multiclass', average='micro', num_classes=len(class_name2idx_dict)).to(device),
            'recall': classification.Recall(task='multiclass', average='none', num_classes=len(class_name2idx_dict)).to(device),
            'micro_avg_recall': classification.Recall(task='multiclass', average='micro', num_classes=len(class_name2idx_dict)).to(device),
            'macro_avg_acc':  classification.Accuracy(task='multiclass', average='macro', num_classes=len(class_name2idx_dict)).to(device),
            'accuracy':  classification.Accuracy(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
            'kappa': classification.CohenKappa(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
            'confusion': classification.ConfusionMatrix(task='multiclass', num_classes=len(class_name2idx_dict)).to(device),
        }
    }
    return metrics_dict
    
def create_and_train_moodel(config_dict: Dict, path_to_saving_dir: str, task:str, crossval_iteration=None):
    t_start = time.time()
    # детерминированные операции и одинаковые зерна генераторов случайных чисел
    if 'deterministic_seed' in config_dict:
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True, warn_only=True)

        set_seed(config_dict['deterministic_seed'])

    if config_dict['segmentation_nn']['params']['encoder_weights'] is None:

        name_postfix = config_dict['name_postfix']
        config_dict['name_postfix'] = f'{name_postfix}_rndw' if len(name_postfix) != 0 else 'rndw'
    # Создание датасета
    if task == 'seismic_sensors':
        train_loader, test_loader, class_name2idx_dict, classes_weights = create_seismic_sensors_dataset(config_dict)
    elif task == 'hsi_uav':
        train_loader, test_loader, class_name2idx_dict, classes_weights = create_hsi_uav_dataset(config_dict)
    # вычисл. устройство, на котором проводится обучение
    device = config_dict['device']

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

    #print(model.encoder)

    # тестовое чтение данных
    for data, labels in train_loader:
        break

    # тестовая обработка данных нейронной сетью
    ret = model(data)
    print('Test model inference:')
    print(f'input_shape:{data.shape}, output_shape:{ret.shape}')

    createion_time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    nn_arch_str = config_dict["segmentation_nn"]["nn_architecture"]
    nn_encoder_str = config_dict["segmentation_nn"]["params"]["encoder_name"]
    name_postfix = config_dict["name_postfix"]
    if crossval_iteration is None:
        if name_postfix is not None and len(name_postfix) != 0:
            model_name = f'{nn_arch_str}_{nn_encoder_str}_{name_postfix} {createion_time_str}'
        else:
            model_name = f'{nn_arch_str}_{nn_encoder_str} {createion_time_str}'
    else:
        if name_postfix is not None and len(name_postfix) != 0:
            model_name = f'{nn_arch_str}_{nn_encoder_str}_{name_postfix}_cv{crossval_iteration} {createion_time_str}'
        else:
            model_name = f'{nn_arch_str}_{nn_encoder_str}_cv{crossval_iteration} {createion_time_str}'

    epoch_num = config_dict['epoch_num']

    print('----------------------------------------------------------')
    print('Created model:')
    print(f'{model_name}')
    print('----------------------------------------------------------')
    print()

    # создаем список словарей с информацией о вычисляемых метриках с помощью multiclass confusion matrix
    # см. подробнее ддокументацию к функции compute_metric_from_confusion
    if task == 'seismic_sensors':
        metrics_dict = create_seismic_metrics(class_name2idx_dict, device)
    elif task == 'hsi_uav':
        metrics_dict = create_hsi_uav_metrics(class_name2idx_dict, device)
    

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
    monitoring_metric = config_dict['monitoring_metric']
    checkpoint_callback = ModelCheckpoint(
        mode="max",
        filename=model_name+f"-{{epoch:02d}}-{{val_{monitoring_metric}:.3}}",
        dirpath=path_to_save_model_dir, 
        save_top_k=1, monitor=f"val_{monitoring_metric}"
        )
    if 'deterministic_seed' in config_dict:
        seed_everything(config_dict['deterministic_seed'])
        #deterministic = 'warn'
        deterministic = False
    else:
        deterministic = False
    trainer = L.Trainer(logger=[csv_logger],
            max_epochs=epoch_num, 
            callbacks=[checkpoint_callback],
            accelerator = 'gpu',
            deterministic = deterministic
            )

    # сохраняем конфигурацию
    path_to_config = os.path.join(path_to_save_model_dir, 'training_config.yaml')
    with open(path_to_config, 'w', encoding='utf-8') as fd:
        #json.dump(config_dict, fd, indent=4)
        yaml.dump(config_dict, fd, indent=4)

    trainer.fit(segmentation_module , train_loader, test_loader,)
    t_stop = time.time()

    elapsed_time = timedelta(seconds=t_stop-t_start)

    print()
    print('----------------------------------------------------------')
    print(f'Training {model_name} is over for {epoch_num} epochs; t={elapsed_time}')
    print('----------------------------------------------------------')
    print()

def search_best_multispecter_bands_combination(config_dict: Dict, path_to_saving_dir:str, basic_bands_indices: List):
    '''
    Выполняется перебор всех возможных комбинаций каналов мультиспектра
    '''
    experiment_date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
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
            create_and_train_moodel(config_dict, path_to_experiment_saving_dir, task='seismic_sensors')
            combination_cnt += 1

def investigate_bands_instride_pretrained(config_dict, path_to_saving_dir, crossval_iteration=None):
    #print(config_dict)
    #print('------------------------------------------------------------------------------------------')
    
    experiment_date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if crossval_iteration is None:
        path_to_experiment_saving_dir = os.path.join(path_to_saving_dir, 'bands_instride_pretrained', f'experiment_{experiment_date}')
        os.makedirs(path_to_experiment_saving_dir, exist_ok=True)
    else:
        path_to_experiment_saving_dir = os.path.join(path_to_saving_dir, 'bands_instride_pretrained_CV')
        os.makedirs(path_to_experiment_saving_dir, exist_ok=True)
    bands_combinations_list = [
        ('b_rgb', [1, 2, 3]),
        ('b_10m', [1, 2, 3, 7]),
        ('b_10-20m', [1, 2, 3, 4, 5, 6, 7, 11, 12]),
        ('b_full_sp', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        #('b_rgb-ndvi', [1, 2, 3, 'ndvi']),
        #('b_rgb-ndwi', [1, 2, 3, 'ndwi']),
        #('b_rgb-ndbi', [1, 2, 3, 'ndbi']),
        #('b_rgb-ndre', [1, 2, 3, 'ndre']),
        #('b_rgb-allind', [1, 2, 3, 'ndvi', 'ndwi', 'ndbi', 'ndre']),
    ]

    inconv_strides_list = {
        ('st_1', (1, 1)),
        ('st_2', (2, 2)),
    }
    pretrained_list = {
        ('w_rnd', None),
        ('w_pr', 'imagenet'),
    }
    all_combinations_list = list(product(bands_combinations_list, inconv_strides_list))
    all_combinations_list = sorted(all_combinations_list, key=lambda x: x[0][0]+x[1][0])
    
    #for i, c in enumerate(all_combinations_list):
    #    print(f'{i+1} {c}')
    #exit()

    all_combinations_list = [{n:v for n, v in entry} for entry in all_combinations_list]
    init_name_postfix = config_dict['name_postfix']
    combinations_num = len(all_combinations_list)
    for combination_cnt, combination in enumerate(all_combinations_list):
        print('#######################################################')
        if crossval_iteration is None:
            print(f'# Train combination #{combination_cnt+1} of total {combinations_num}')
        else:
            print(f'# Cross-validation Iteration #{crossval_iteration+1}')
            print(f'# Train combination #{combination_cnt+1} of total {combinations_num}')
        print('#######################################################')
        name_postfix = init_name_postfix
        for comb_name, comb_val in combination.items():
            if comb_name.startswith('b_'):
                config_dict['multispecter_bands_indices'] = comb_val
            elif comb_name.startswith('st_'):
                config_dict['segmentation_nn']['input_layer_config']['params']['stride'] = comb_val
            elif comb_name.startswith('w_'):
                config_dict['segmentation_nn']['params']['encoder_weights'] = comb_val
            name_postfix = f'{name_postfix}_{comb_name}'

        config_dict['name_postfix'] = name_postfix

        create_and_train_moodel(config_dict, path_to_experiment_saving_dir, task='seismic_sensors', crossval_iteration=crossval_iteration)

        config_dict['name_postfix'] = init_name_postfix

def seismic_5x2cross_val(cv_config_dict, path_to_saving_dir):
    for i in range(5):
        
        cv_config_dict['crossval_iteration'] = i
        investigate_bands_instride_pretrained(cv_config_dict, path_to_saving_dir, crossval_iteration=i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_to_model_configs', nargs='+')
    parser.add_argument('--paths_to_encoder_configs', nargs='+')
    parser.add_argument('--training_mode', help='Mode of training. Available options: "single_nn", "search_best_bands", "investigate_bands_instride", "crossval_bands_instride"')
    parser.add_argument('--path_to_saving_dir')
    parser.add_argument('--task', help='Could be `hsi_uav` OR `seismic_sensors`')

    sample_args = [
        '--paths_to_model_configs',
        #'training_configs/models/unet++.yaml',
        #'training_configs/models/fpn.yaml',
        #'training_configs/models/fcn.yaml',
        #'training_configs/models/fcn1.yaml',
        'training_configs/models/unet.yaml',
        #'training_configs/models/unet_hsi.yaml',
        #'training_configs/models/unet_aux_chtr_hsi.yaml',
        #'training_configs/models/unet_aux_patr_hsi.yaml',

        '--paths_to_encoder_configs',
        #'training_configs/encoders/tu-maxvit_tiny.yaml',
        'training_configs/encoders/efficientnet-b2.yaml',
        #'training_configs/encoders/tu-cspdarknet53.yaml',
        #'training_configs/encoders/tu-mobilenetv4_hybrid_medium.yaml',
        #'training_configs/encoders/densenet121.yaml',
        #'training_configs/encoders/tu-seresnext50_32x4d.yaml',
        
        '--training_mode', 'crossval_bands_instride',
        '--path_to_saving_dir', 'saving_dir',
        '--task', 'seismic_sensors'

    ]
    args = parser.parse_args(sample_args)
    #print(args)
    paths_to_model_configs = args.paths_to_model_configs
    paths_to_encoder_configs = args.paths_to_encoder_configs
    training_mode = args.training_mode
    path_to_saving_dir = args.path_to_saving_dir
    task = args.task
    
    if training_mode == 'single_nn':
        
        for path_to_model_config in paths_to_model_configs:
            with open(path_to_model_config) as fd:
                if path_to_model_config.endswith('.yaml'):
                    config_dict = yaml.load(fd, Loader=yaml.Loader)
                elif path_to_model_config.endswith('.json'):
                    config_dict = json.load(fd)

            for path_to_encoder_config in paths_to_encoder_configs:
                current_model_config = deepcopy(config_dict)
                with open(path_to_encoder_config) as fd:
                    if path_to_encoder_config.endswith('.yaml'):
                        encoder_dict = yaml.load(fd, Loader=yaml.Loader)
                    elif path_to_encoder_config.endswith('.json'):
                        encoder_dict = json.load(fd)

                current_model_config['segmentation_nn']['input_layer_config'] = encoder_dict['input_layer_config']

                current_model_config['segmentation_nn']['params'].update(encoder_dict['model_params'])
                create_and_train_moodel(current_model_config, path_to_saving_dir, task=task)

    elif training_mode in ('investigate_bands_instride', 'crossval_bands_instride'):
        for path_to_model_config in paths_to_model_configs:
            with open(path_to_model_config) as fd:
                if path_to_model_config.endswith('.yaml'):
                    config_dict = yaml.load(fd, Loader=yaml.Loader)
                elif path_to_model_config.endswith('.json'):
                    config_dict = json.load(fd)

            
            for path_to_encoder_config in paths_to_encoder_configs:
                current_model_config = deepcopy(config_dict)
                with open(path_to_encoder_config) as fd:
                    if path_to_encoder_config.endswith('.yaml'):
                        encoder_dict = yaml.load(fd, Loader=yaml.Loader)
                    elif path_to_encoder_config.endswith('.json'):
                        encoder_dict = json.load(fd)

                current_model_config['segmentation_nn']['input_layer_config'] = encoder_dict['input_layer_config']

                current_model_config['segmentation_nn']['params'].update(encoder_dict['model_params'])
                if training_mode == 'crossval_bands_instride':
                    seismic_5x2cross_val(current_model_config, path_to_saving_dir)
                elif training_mode == 'investigate_bands_instride':
                    investigate_bands_instride_pretrained(current_model_config, path_to_saving_dir)

    elif training_mode == 'search_best_bands':
        path_to_config = paths_to_model_configs[0]
        # чтение файла конфигурации
        with open(path_to_config) as fd:
            if path_to_config.endswith('.yaml'):
                config_dict = yaml.load(fd, Loader=yaml.Loader)
            elif path_to_config.endswith('.json'):
                config_dict = json.load(fd)
        search_best_multispecter_bands_combination(config_dict, path_to_saving_dir, basic_bands_indices=[1, 2, 3, 7])


    

