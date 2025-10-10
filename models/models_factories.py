import torch
from torch import nn

import glob
import os
from tqdm import tqdm
from datetime import datetime
import json
import yaml

from argparse import ArgumentParser

from itertools import combinations
import warnings

import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision import models

import torch.nn.functional as F

import segmentation_models_pytorch as smp

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from typing import Dict

from .fcn import FCN
from .fcn1 import FCN1
from .fpn import FPNMod
from .unet_pp import UnetPlusPlusMod
from .att_unet import UnetAtt, ConcatDim1, ConvCrossAttentionBlock, ConvMSABlock

from .losses import DiceCELoss

class MultisizeConv(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:dict,
            kernel_size:dict,
            stride:dict,
            padding:dict,
            dilation:dict,
            groups:dict,
            bias:dict,
            aggregation_type:str,
            ):
        '''
        Класс для замены входного слоя на некий аналог Inception модуля
        для обеспечения
        '''
        super().__init__()
        
        self.aggregation_type = aggregation_type
        self.multisize_convs = nn.ModuleDict()
        for conv_name in kernel_size.keys():
            self.multisize_convs[conv_name] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels[conv_name],
                kernel_size=kernel_size[conv_name],
                stride=stride[conv_name],
                padding=padding[conv_name],
                dilation=dilation[conv_name],
                groups=groups[conv_name],
                bias=bias[conv_name]
                )
        
    def update_weights(self, new_weights_dict):
        '''
        На вход принимается словрь со структурой {'имя_свертки': (weight, bias)}
        '''
        for conv_name, (weight, bias) in new_weights_dict.items():
            self.multisize_convs[conv_name].weight = nn.Parameter(weight)
            if self.multisize_convs[conv_name].bias is not None:
                self.multisize_convs[conv_name].bias = nn.Parameter(bias)

    def forward(self, x):
        outputs = []
        for conv_name in self.multisize_convs.keys():
            out = self.multisize_convs[conv_name](x)
            #print(out.shape)
            outputs.append(out)
        
        if self.aggregation_type == 'add':
            outputs = torch.stack(outputs, dim=0)
            outputs = outputs.sum(dim=0)
        elif self.aggregation_type == 'cat':
            outputs = torch.cat(outputs, dim=1)
        else:
            raise ValueError(f'self.aggregation_type should be either "add" or "cat". Got {self.aggregation_type}')
        return outputs

transforms_factory_dict = {
    'affine': v2.RandomAffine,
    'perspective': v2.RandomPerspective,
    'horizontal_flip': v2.RandomHorizontalFlip,
    'vertical_flip': v2.RandomVerticalFlip,
    'crop': v2.RandomCrop,
    'gauss_noise': v2.GaussianNoise,
    'gauss_blur': v2.GaussianBlur,
    'elastic': v2.ElasticTransform,
}

segmentation_nns_factory_dict = {
    'unet': smp.Unet,
    'att_unet': UnetAtt,
    'fpn': smp.FPN,
    'custom_fpn': FPNMod,
    'unet++': UnetPlusPlusMod,
    'fcn': FCN,
    'fcn1': FCN1,
    #'custom_manet': MAnetMod,
}

criterion_factory_dict = {
    'crossentropy': nn.CrossEntropyLoss,
    'dice_crossentropy': DiceCELoss,
    'dice': smp.losses.DiceLoss
}

optimizers_factory_dict = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}

lr_schedulers_factory_dict = {
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
}    

def create_weights_from_avg_ch(weight, new_in_channels):
    return torch.cat([weight.mean(dim=1).unsqueeze(1)]*new_in_channels, dim=1)

def cerate_weights_from_repeated_ch(weight, in_channels, new_in_channels):
    ch_multiple = new_in_channels//in_channels
    reminded_channels = new_in_channels%in_channels
    # сначала набираем новые каналы путем подставления друг за другом (stack) каналов изначального изображения,
    # а затем, если количество новых каналов не делится без остатка на количество изначальных, 
    # то набираем оставшиеся новые каналы из оставшихся изначальных    
    new_weight = torch.cat(
        [weight]*ch_multiple + [weight[:,:reminded_channels]], dim=1)
    return new_weight

def create_augmentation_transforms(transforms_dict:Dict[str, Dict]):
    transforms_list = []
    for name, transform_params in transforms_dict.items():
        transform_creation_fn = transforms_factory_dict[name]
        transforms_list.append(transform_creation_fn(**transform_params))
    #return v2.Compose([v2.RandomOrder(transforms_list)])
    return v2.RandomOrder(transforms_list)

def create_model(config_dict, segmentation_nns_factory_dict):
    model_name = config_dict['segmentation_nn']['nn_architecture']
    if 'fpn' in model_name:
        stride = config_dict['segmentation_nn']['input_layer_config']['params']['stride']
        if isinstance(stride, (list, tuple)):
            stride_val = stride[0]
        elif isinstance(stride, (list, tuple)):
            stride_val = stride
        #if stride_val != 1:
        config_dict['segmentation_nn']['params']['upsampling'] = stride_val
    # создаем нейронную сеть из фабрики
    model = segmentation_nns_factory_dict[model_name](**config_dict['segmentation_nn']['params'])
    multispecter_bands_indices = config_dict['multispecter_bands_indices']
    in_channels = len(multispecter_bands_indices)
    
    
    input_conv = model.get_submodule(
        config_dict['segmentation_nn']['input_layer_config']['layer_path']
        )
    if 'channels' in config_dict['segmentation_nn']['input_layer_config']['replace_type']:
        if 'cspdarknet' in config_dict['segmentation_nn']['params']['encoder_name']:
            new_stride = input_conv.stride
            new_pad = input_conv.padding

            repl_stride_conv = model.get_submodule(
                config_dict['segmentation_nn']['input_layer_config']['stride_repl_layer_path']
                )
            repl_stride_conv.stride = config_dict['segmentation_nn']['input_layer_config']['params']['stride']
            repl_stride_conv.padding = config_dict['segmentation_nn']['input_layer_config']['params']['padding']

            model.set_submodule(
                config_dict['segmentation_nn']['input_layer_config']['stride_repl_layer_path'],
                repl_stride_conv
                )

        else:
            new_stride = config_dict['segmentation_nn']['input_layer_config']['params']['stride']
            new_pad = config_dict['segmentation_nn']['input_layer_config']['params']['padding']
        new_input_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=input_conv.out_channels,
            kernel_size=input_conv.kernel_size,
            stride=new_stride,
            padding=new_pad,
            dilation=input_conv.dilation,
            groups=input_conv.groups,
            bias=input_conv.bias is not None
        )
        if in_channels != 3:
            # получаем входной слой, специфический для конкретной нейронной сети
            
            
            if config_dict['segmentation_nn']['params']['encoder_weights'] is not None:
                # выбор типа обнолвления весов
                if config_dict['segmentation_nn']['input_layer_config']['weight_update_type'] == 'average_all':
                    
                    #new_weight = torch.cat([input_conv.weight.mean(dim=1).unsqueeze(1)]*in_channels, dim=1)
                    new_weight = create_weights_from_avg_ch(input_conv.weight, in_channels)
                    input_conv.weight = nn.Parameter(new_weight)

                elif config_dict['segmentation_nn']['input_layer_config']['weight_update_type'] == 'repeate':
                    '''
                    ch_multiple = in_channels//input_conv.in_channels
                    reminded_channels = in_channels%input_conv.in_channels
                    new_weight = torch.cat(
                        [input_conv.weight]*ch_multiple + [input_conv.weight[:,:reminded_channels]], dim=1)
                    '''
                    new_weight = cerate_weights_from_repeated_ch(input_conv.weight, input_conv.in_channels, in_channels)
                    new_input_conv.weight = nn.Parameter(new_weight)
        else:
            # если у нас три канала на входе, то просто перезаписываем вес
            new_input_conv.weight = nn.Parameter(input_conv.weight)

        if input_conv.bias is not None:
            new_input_conv.bias = input_conv.bias
        # перезаписываем входной слой исходя из специфики оригинальной сети
        model.set_submodule(
                config_dict['segmentation_nn']['input_layer_config']['layer_path'],
                new_input_conv
                )

    elif 'multisize_conv' in config_dict['segmentation_nn']['input_layer_config']['replace_type']:
        multisize_params = config_dict['segmentation_nn']['input_layer_config']['params']
        new_input_conv = MultisizeConv(**multisize_params)

        # Если мы модифицируем входной слой.
        if config_dict['segmentation_nn']['params']['encoder_weights'] is not None:
            # вычленяем словрь с параметрами размеров ядер сверток.
            kernel_sizes_dict = config_dict['segmentation_nn']['input_layer_config']['params']['kernel_size']
            interpolated_kernels_dict = {}
            # выполняем интерполяцию ядер свертки для каждого набора из новых ядер
            for name, kernel_size in kernel_sizes_dict.items():
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                # получаем интерполированную версию ядер свертки
                interpolated_kernels_dict[name] = [
                    F.interpolate(input_conv.weight, size=kernel_size, mode='bicubic', antialias=True),
                    input_conv.bias]
                '''            
                out_channels_dict = config_dict['segmentation_nn']['input_layer_config']['params']['out_channels']
                for name, out_channels in out_channels_dict.items():
                    weights = interpolated_kernels_dict[name][0]
                    weights = create_weights_from_avg_ch(weights, in_channels)
                    interpolated_kernels_dict[name][0] = weights
                '''
            #out_channels_dict = config_dict['segmentation_nn']['input_layer_config']['params']['out_channels']
            for name in interpolated_kernels_dict.keys():
                weights = interpolated_kernels_dict[name][0]
                if config_dict['segmentation_nn']['input_layer_config']['weight_update_type'] == 'average_all':
                    weights = create_weights_from_avg_ch(weights, new_in_channels=in_channels)
                elif config_dict['segmentation_nn']['input_layer_config']['weight_update_type'] == 'repeat':
                    weights = cerate_weights_from_repeated_ch(weights, in_channels=input_conv.in_channels, new_in_channels=in_channels)
                
                interpolated_kernels_dict[name][0] = weights
                        
            new_input_conv.update_weights(new_weights_dict=interpolated_kernels_dict)
        if config_dict['segmentation_nn']['input_layer_config']['params']['aggregation_type'] == 'cat':
            # Если тип агрегации выхода MultisizeConv - это конкатенация, то изменяем также второй сверточный слой,
            # чтобы число его входных каналов соответствовало числу выходных первого слоя 
            raise NotImplementedError
        # заменяем сходной слой по заранее определенному пути, который может варьировать в зависимости от архитектуры энкодера
        model.set_submodule(
                config_dict['segmentation_nn']['input_layer_config']['layer_path'],
                new_input_conv
                )
    return model