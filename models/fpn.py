import torch
from torch import nn

import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from typing import Any, Optional, List, Literal

class FPNMod(SegmentationModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_pyramid_channels: A number of convolution filters in Feature Pyramid of FPN_
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks of FPN_
        decoder_merge_policy: Determines how to merge pyramid features inside FPN. Available options are **add**
            and **cat**
        decoder_dropout: Spatial dropout rate in range (0, 1) for feature pyramid in FPN_
        decoder_interpolation: Interpolation mode used in decoder of the model. Available options are
            **"nearest"**, **"bilinear"**, **"bicubic"**, **"area"**, **"nearest-exact"**. Default is **"nearest"**.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_type:str = 'conv',
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        # validate input params
        if encoder_name.startswith("mit_b") and encoder_depth != 5:
            raise ValueError(
                "Encoder {} support only encoder_depth=5".format(encoder_name)
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = FPNDecoderMod(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
            interpolation_mode=decoder_interpolation,
            encoder_type=encoder_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

class FPNModBlock(nn.Module):
    def __init__(
        self,
        pyramid_channels: int,
        skip_channels: int,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.interpolation_mode = interpolation_mode

    def forward(self, x: torch.Tensor, skip: torch.Tensor, scale_factor: float) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=scale_factor, mode=self.interpolation_mode)
        if skip.size(1) != 0:
            #print(x.shape, skip.shape)
            skip = self.skip_conv(skip)
            x = x + skip
        return x

class FPNDecoderMod(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        encoder_depth: int = 5,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        dropout: float = 0.2,
        merge_policy: Literal["add", "cat"] = "add",
        interpolation_mode: str = "nearest",
        encoder_type:str = 'conv',
    ):
        super().__init__()

        self.out_channels = (
            segmentation_channels
            if merge_policy == "add"
            else segmentation_channels * 4
        )
        #print(self.out_channels)
        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for FPN decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]
        
        self.p6 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        '''
        self.p5 = smp.decoders.fpn.decoder.FPNBlock(pyramid_channels, encoder_channels[1], interpolation_mode)
        self.p4 = smp.decoders.fpn.decoder.FPNBlock(pyramid_channels, encoder_channels[2], interpolation_mode)
        self.p3 = smp.decoders.fpn.decoder.FPNBlock(pyramid_channels, encoder_channels[3], interpolation_mode)
        self.p2 = smp.decoders.fpn.decoder.FPNBlock(pyramid_channels, encoder_channels[4], interpolation_mode)
        '''
        self.p5 = FPNModBlock(pyramid_channels, encoder_channels[1], interpolation_mode)
        self.p4 = FPNModBlock(pyramid_channels, encoder_channels[2], interpolation_mode)
        self.p3 = FPNModBlock(pyramid_channels, encoder_channels[3], interpolation_mode)
        self.p2 = FPNModBlock(pyramid_channels, encoder_channels[4], interpolation_mode)
        
        if encoder_type == 'conv':
            upsamples_list = [4, 3, 2, 1, 0]
        elif encoder_type == 'vit':
            upsamples_list = [3, 2, 1, 0, 0]

        self.seg_blocks = nn.ModuleList(
            [
                smp.decoders.fpn.decoder.SegmentationBlock(
                    pyramid_channels, segmentation_channels, n_upsamples=n_upsamples
                )
                for n_upsamples in upsamples_list
            ]
        )

        self.merge = smp.decoders.fpn.decoder.MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5, c6 = features[-5:]

        p6 = self.p6(c6)
        p5 = self.p5(p6, c5, scale_factor=2.0)
        p4 = self.p4(p5, c4, scale_factor=2.0)
        p3 = self.p3(p4, c3, scale_factor=2.0)
        p2 = self.p2(p3, c2, scale_factor=2.0)

        s6 = self.seg_blocks[0](p6)
        s5 = self.seg_blocks[1](p5)
        s4 = self.seg_blocks[2](p4)
        s3 = self.seg_blocks[3](p3)
        s2 = self.seg_blocks[4](p2)

        feature_pyramid = [s6, s5, s4, s3, s2]

        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        
        return x