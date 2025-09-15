import torch
from torch import nn

import segmentation_models_pytorch as smp

from typing import List

class DiceCELoss(nn.Module):
    def __init__(
            self,
            ce_weight,
            ce_ignore_index,
            ce_reducion,
            ce_label_smoothing,
            dice_mode,
            dice_classes,
            dice_log_loss,
            dice_from_logits,
            dice_smooth,
            dice_ignore_index,
            dice_eps,
            losses_weight: List = [0.5, 0.5],
            is_trainable_weights: bool = False,
            weights_processing_type: str = None,
            ):
        super().__init__()
        self.dice = smp.losses.DiceLoss(
            mode=dice_mode,
            classes=dice_classes,
            log_loss=dice_log_loss,
            from_logits=dice_from_logits,
            smooth=dice_smooth,
            ignore_index=dice_ignore_index,
            eps=dice_eps
            )
        self.ce = nn.CrossEntropyLoss(
            weight=ce_weight,
            ignore_index=ce_ignore_index,
            reduction=ce_reducion,
            label_smoothing=ce_label_smoothing,
        )
        self.loss_weights = torch.tensor(losses_weight)
        if is_trainable_weights:
            self.loss_weights = nn.Parameter(self.loss_weights)
        self.weights_processing_type = weights_processing_type

    def forward(self, pred, true):
        weights = self.loss_weights
        if self.weights_processing_type == 'softmax':
            weights = weights.softmax(dim=0)
        elif self.weights_processing_type == 'sigmoid':
            weights = weights.softmax(dim=0)

        ce_loss = self.ce(pred, true) * weights[0]
        dice_loss = self.dice(pred, true) * weights[1]
        return ce_loss + dice_loss