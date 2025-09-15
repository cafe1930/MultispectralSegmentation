import torch
from torch import nn

import lightning as L

def compute_pred_mask(pred):
    '''
    Определение маски классов на основе сгенерированной softmax маски
    '''
    #pred = pred.detach()
    _, pred_mask = pred.max(dim=1)
    return pred_mask#.cpu().numpy()

class LightningSegmentationModule(L.LightningModule):
    def __init__(self, model:nn.Module, criterion:nn.Module, optimizer_cfg:dict, metrics_dict:dict, name2class_idx_dict:dict) -> None:
        '''
        Модуль Lightning для обучения сегментационной сети
        In:
            model - нейронная сеть
            criterion - функция потерь
            
            name2class_idx_dict - словарь с отображением {class_name(str): class_idx(int)}
        '''
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_cfg = optimizer_cfg
        self.metrics_dict = metrics_dict
        
        self.name2class_idx_dict = name2class_idx_dict
        # словарь, выполняющий обратное отображение class_idx в class_name
        self.class_idx2name_dict = {v:k for k, v in name2class_idx_dict.items()}
        
    def configure_optimizers(self):
        optimizer = self.optimizer_cfg['optmizer'](self.parameters(), **self.optimizer_cfg['optimizer_args'])
        ret_dict = {'optimizer': optimizer}
        if self.optimizer_cfg['lr_scheduler'] is not None:
            scheduler = self.optimizer_cfg['lr_scheduler'](optimizer, **self.optimizer_cfg['lr_scheduler_args'])
            ret_dict['lr_scheduler'] = {'scheduler': scheduler}
            ret_dict['lr_scheduler'].update(self.optimizer_cfg['lr_scheduler_params'])
        
        return ret_dict

    def compute_metrics(self, pred_labels, true_labels, mode):
        metrics_names_list = self.metrics_dict[mode].keys()
        for metric_name in metrics_names_list:
            if 'dice' in metric_name.lower():
                self.metrics_dict[mode][metric_name].update(pred_labels, true_labels)
            else:
                self.metrics_dict[mode][metric_name].update(pred_labels.reshape(-1), true_labels.reshape(-1))
        
    
    def training_step(self, batch, batch_idx):
        data, true_labels = batch
        pred = self.model(data)
        loss = self.criterion(pred, true_labels)
        # вычисление сгенерированной маски
        pred_labels = compute_pred_mask(pred)
        #true_labels = true_labels.detach().cpu().numpy()
        
        self.compute_metrics(pred_labels=pred_labels, true_labels=true_labels, mode='train')

        # т.к. мы вычисляем общую ошибку на всей эпохе, то записываем в лог только значение функции потерь
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, true_labels = batch
        pred = self.model(data)
        loss = self.criterion(pred, true_labels)
        pred_labels = compute_pred_mask(pred)
        self.compute_metrics(pred_labels=pred_labels, true_labels=true_labels, mode='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def log_metrics(self, mode):
        for metric_name, metric in self.metrics_dict[mode].items():
            metric_val = metric.compute()
            if 'confusion' in metric_name.lower():
                disp_name = f'{mode}_{metric_name}'
                self.log(disp_name, metric_val.cpu().tolist(), on_step=False, on_epoch=True, prog_bar=False)
            else:
                for i, value in enumerate(metric_val):
                    class_name = self.class_idx2name_dict[i]
                    disp_name = f'{mode}_{metric_name}_{class_name}'
                    self.log(disp_name, value, on_step=False, on_epoch=True, prog_bar=True)
                disp_name = f'{mode}_{metric_name}_mean'
                self.log(disp_name, metric_val.mean(), on_step=False, on_epoch=True, prog_bar=True)
            self.metrics_dict[mode][metric_name].reset()

    def on_train_epoch_end(self):
        '''
        Декодирование результатов тренировочной эпохи и запись их в лог
        '''
        self.log_metrics(mode='train')
 
    def on_validation_epoch_end(self):
        '''
        Декодирование результатов тестовой эпохи и запись их в лог
        (работает точно также, как и )
        '''
        self.log_metrics(mode='val')


