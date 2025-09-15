import torch

import os


import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors

import warnings

import numpy as np

import pandas as pd

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset_root:str, samples_df:pd.DataFrame, channel_indices:list, transforms:v2._transform.Transform, dtype:torch.dtype, device:torch.device):
        '''
        In:
            path_to_dataset_root - путь до корневой папки с датасетом
            samples_df - pandas.DataFrame с информацией о файлах
            channel_indices - список с номерами каналов мультиспектрального изображения
            transforms - аугментация изображений
        '''
        super().__init__()

        self.path_to_dataset_root = path_to_dataset_root
        self.samples_df = samples_df
        self.specter_bands_list = [i for i in channel_indices if isinstance(i, int)]
        self.specter_indices_names = [s for s in channel_indices if isinstance(s, str)]
        self.dtype_trasform = v2.ToDtype(dtype=dtype, scale=True)
        self.other_transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.samples_df)
    @staticmethod
    def compute_spectral_index(index_name, image):
        if index_name.lower() == 'ndvi':
            b0 = image[7] # NIR, B8
            b1 = image[3] # RED, B4
            
        elif index_name.lower() == 'ndbi':
            b0 = image[10] #SWIR, B11
            b1 = image[7] #NIR, B8

        elif index_name.lower() == 'ndwi':
            b0 = image[2] #green, B3
            b1 = image[7] #NIR, B8

        elif index_name.lower() == 'ndre':
            b0 = image[7] #NIR, B8
            b1 = image[5] #Red Edge, B6
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            index = (b0 - b1)/(b0 + b1)
            
        index = np.nan_to_num(index, nan=-5)

        return index
            

    def __getitem__(self, idx):
        sample = self.samples_df.iloc[idx]

        file_name = sample['file_name']

        path_to_image = os.path.join(self.path_to_dataset_root, 'images', f'{file_name}.npy')
        path_to_labels = os.path.join(self.path_to_dataset_root, 'labels', f'{file_name}.npy')

        image = np.load(path_to_image)
        spectral_indices = []
        # вычисляем спектральные индексы
        if len(self.specter_indices_names) > 0:
            for sp_index_name in self.specter_indices_names:
                spectral_index = self.compute_spectral_index(sp_index_name, image)
                spectral_index = torch.as_tensor(spectral_index)
                spectral_indices.append(spectral_index.unsqueeze(0))

            spectral_indices = torch.cat(spectral_indices)
            spectral_indices = self.dtype_trasform(spectral_indices)


        image = torch.as_tensor(image[self.specter_bands_list], dtype=torch.int16)
        image = self.dtype_trasform(image)
        # добавляем спектральные индексы
        if len(self.specter_indices_names) > 0:
            image = torch.cat([image, spectral_indices], dim=0) 
        #image = np.load(path_to_image)
        # метки читаем как одноканальное изображение
        label = np.load(path_to_labels)
        label = np.where(label >= 0, label, 0)
        #label = torch.as_tensor(np.load(path_to_labels), dtype=torch.uint8).long()
        label = torch.as_tensor(label, dtype=torch.uint8).long()
        
        image = tv_tensors.Image(image, device=self.device)
        label = tv_tensors.Mask(label, device=self.device)

        transforms_dict = {'image':image, 'mask':label}
        transformed = self.other_transforms(transforms_dict)
        return transformed['image'], transformed['mask']#, image