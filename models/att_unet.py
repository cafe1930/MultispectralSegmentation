import torch
from torch import nn
import warnings
import torchvision
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from typing import Any, Dict, Optional, Union, Callable, Sequence, List
import einops as eo

def scaled_dot_product_attention(query, key, value, mask=None):
    # query, key, value are (batch_size, num_heads, seq_len, head_dim)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # Apply mask for padding or causal attention

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class ConvMultiheadAttention(nn.Module):
    '''
    Class for convolutional multihead self-attention Wq, Wk, Wv are replaced from fully connected to convolutional layers 
    '''
    def __init__(
            self,
            q_in_channels,
            k_in_channels,
            v_in_channels,
            q_out_channels,
            k_out_channels,
            v_out_channels,
            in_kernel_size,
            in_padding,
            in_stride,
            head_row_dim,
            head_col_dim,
            head_ch_dim,
            norm=nn.LayerNorm
            ):
        super().__init__()

        self.head_row_dim = head_row_dim
        self.head_col_dim = head_col_dim
        self.head_ch_dim = head_ch_dim

        self.input_norm = norm(head_ch_dim*head_row_dim*head_col_dim, eps=1e-6)
        
        self.conv_q = nn.Conv2d(
            in_channels=q_in_channels,
            out_channels=q_out_channels,
            kernel_size=in_kernel_size,
            padding=in_padding,
            stride=in_stride
            )
        
        self.conv_k = nn.Conv2d(
            in_channels=k_in_channels,
            out_channels=k_out_channels,
            kernel_size=in_kernel_size,
            padding=in_padding,
            stride=in_stride
            )
        
        self.conv_v = nn.Conv2d(
            in_channels=v_in_channels,
            out_channels=v_out_channels,
            kernel_size=in_kernel_size,
            padding=in_padding,
            stride=in_stride
            )
        
        #self.out_conv = nn.Conv2d(in_channels=q_out_channels,out_channels=out_channels,kernel_size=out_kernel_size,padding=out_padding,stride=out_stride)
        
    def forward(self, query, key, value, mask=None):
        bs, ch, rows, cols = value.shape
        head_row_num = rows // self.head_row_dim
        head_col_num = cols // self.head_col_dim
        head_ch_num = ch // self.head_ch_dim

        #print(head_row_num, head_col_num)

        q = self.conv_q(query)
        k = self.conv_k(key)
        v = self.conv_v(value)
        qkv_rearrangement_str = 'bs (head_ch_dim head_ch_num) (head_rdim head_rnum) (head_cdim head_cnum) -> bs (head_rnum head_cnum head_ch_num) (head_rdim head_cdim head_ch_dim) '
        
        q = eo.rearrange(
            q,
            qkv_rearrangement_str,
            head_rdim=self.head_row_dim, head_cdim=self.head_col_dim, head_ch_dim=self.head_ch_dim)

        k = eo.rearrange(
            k,
            qkv_rearrangement_str,
            head_rdim=self.head_row_dim, head_cdim=self.head_col_dim, head_ch_dim=self.head_ch_dim
            )
        
        v = eo.rearrange(
            v,
            qkv_rearrangement_str,
            head_rdim=self.head_row_dim, head_cdim=self.head_col_dim, head_ch_dim=self.head_ch_dim)
        #print(f'q:{q.shape};k:{k.shape};v:{v.shape}')
        
        weighted_v = F.scaled_dot_product_attention(query=q, key=k, value=v)
        #print(f'v_w:{weighted_v.shape}')

        weighted_v = eo.rearrange(
            weighted_v,
            'bs (head_rnum head_cnum head_ch_num) (head_rdim head_cdim head_ch_dim) -> bs (head_ch_dim head_ch_num) (head_rdim head_rnum) (head_cdim head_cnum)',
            head_rdim=self.head_row_dim, head_cdim=self.head_col_dim, head_rnum=head_row_num, head_cnum=head_col_num,
        )
        #print(f'v_w_ra:{weighted_v.shape}')
        return weighted_v

        out = self.out_conv(weighted_v)
        return out
    
class ConvMSABlock(nn.Module):
    '''
    Implementatuion of convolutional multihead self-attention block
    '''
    def __init__(
            self,
            msa_in_channels,
            msa_intermediate_channels,
            msa_in_kernel_size,
            msa_in_padding,
            msa_in_stride,
            msa_out_channels,
            
            msa_head_row_dim,
            msa_head_col_dim,
            msa_head_ch_dim,

            dropout,

            out_conv_hidden_channels,
            out_conv_kernel_size,
            out_conv_padding,
            out_conv_stride,

            out_conv_out_channels,

            out_conv_act,
            
            norm_layer: Callable[..., torch.nn.Module],
            ):
        super().__init__()
        
        self.norm1 = norm_layer(msa_in_channels)
        self.self_att = ConvMultiheadAttention(
            q_in_channels=msa_in_channels,
            k_in_channels=msa_in_channels,
            v_in_channels=msa_in_channels,
            q_out_channels=msa_intermediate_channels,
            k_out_channels=msa_intermediate_channels,
            v_out_channels=msa_intermediate_channels,
            in_kernel_size=msa_in_kernel_size,
            in_padding=msa_in_padding,
            in_stride=msa_in_stride,
            head_row_dim=msa_head_row_dim,
            head_col_dim=msa_head_col_dim,
            head_ch_dim=msa_head_ch_dim,
        )

        self.dropout = nn.Dropout2d(dropout)

        self.out_conv = nn.Sequential(
            torchvision.ops.Conv2dNormActivation(
                in_channels=msa_out_channels,
                out_channels=out_conv_hidden_channels,
                kernel_size=out_conv_kernel_size,
                padding=out_conv_padding,
                stride=out_conv_stride,
                activation_layer=out_conv_act
            ),
            torchvision.ops.Conv2dNormActivation(
                in_channels=out_conv_hidden_channels,
                out_channels=out_conv_out_channels,
                kernel_size=out_conv_kernel_size,
                padding=out_conv_padding,
                stride=out_conv_stride,
                activation_layer=out_conv_act
            )
        )

        self.norm2 = norm_layer(msa_out_channels)
    def forward(self, input_features):
        
        x = self.norm1(input_features)
        #print(x.shape)
        
        x = self.self_att(query=x, key=x, value=x)
        
        x = self.dropout(x)
        x = x + input_features
        x = self.norm2(x)
        #print(x.shape)
        y = self.out_conv(x)
        return x + y
    

class ConvCrossAttentionBlock(nn.Module):
    '''
    Implementation of convolutional cross-attention block
    '''
    def __init__(
            self,
            q_in_channels,
            k_in_channels,
            v_in_channels,
            q_out_channels,
            k_out_channels,
            v_out_channels,
            in_kernel_size,
            in_padding,
            in_stride,
            
            head_row_dim,
            head_col_dim,
            head_ch_dim,

            dropout,
            
            norm_layer: Callable[..., torch.nn.Module],
            ):
        super().__init__()
        
        self.kv_inp_norm = norm_layer(k_in_channels)
        self.q_inp_norm = norm_layer(q_in_channels)

        self.cross_att = ConvMultiheadAttention(
            q_in_channels=q_in_channels,
            k_in_channels=k_in_channels,
            v_in_channels=v_in_channels,
            q_out_channels=q_out_channels,
            k_out_channels=k_out_channels,
            v_out_channels=v_out_channels,
            in_kernel_size=in_kernel_size,
            in_padding=in_padding,
            in_stride=in_stride,
            
            head_row_dim=head_row_dim,
            head_col_dim=head_col_dim,
            head_ch_dim=head_ch_dim,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.out_norm = norm_layer(q_out_channels)

    def forward(self, q, kv):
        
        x = self.kv_inp_norm(kv)
        q = self.q_inp_norm(q)
        #print(x.shape)
        
        x = self.cross_att(query=q, key=x, value=x)
        
        x = self.dropout(x)
        #print(x.shape, q.shape)
        x = x + q
        x = self.out_norm(x)
        
        return x
    
class ConcatDim1(nn.Module):
    '''
    Implementation of concatenation. It is nececcary for various aggreagation strategies in UNet decoder
    '''
    def forward(self, *tensors):
        return torch.cat(tensors, dim=1)
    

unet_aggregation_factory_dict = {
    'concat': ConcatDim1,
    'conv_cross_att': ConvCrossAttentionBlock,
}

unet_attention_factory_dict = {
    'conv_msa': ConvMSABlock,
    'none': nn.Identity,
}
    

class UnetAtt(SegmentationModel):
    """
    U-Net is a fully convolutional neural network architecture designed for semantic image segmentation.

    It consists of two main parts:

    1. An encoder (downsampling path) that extracts increasingly abstract features
    2. A decoder (upsampling path) that gradually recovers spatial details

    The key is the use of skip connections between corresponding encoder and decoder layers.
    These connections allow the decoder to access fine-grained details from earlier encoder layers,
    which helps produce more precise segmentation masks.

    The skip connections work by concatenating feature maps from the encoder directly into the decoder
    at corresponding resolutions. This helps preserve important spatial information that would
    otherwise be lost during the encoding process.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_norm:     Specifies normalization between Conv2D and activation.
            Accepts the following types:
            - **True**: Defaults to `"batchnorm"`.
            - **False**: No normalization (`nn.Identity`).
            - **str**: Specifies normalization type using default parameters. Available values:
              `"batchnorm"`, `"identity"`, `"layernorm"`, `"instancenorm"`, `"inplace"`.
            - **dict**: Fully customizable normalization settings. Structure:
              ```python
              {"type": <norm_type>, **kwargs}
              ```
              where `norm_name` corresponds to normalization type (see above), and `kwargs` are passed directly to the normalization layer as defined in PyTorch documentation.

            **Example**:
            ```python
            decoder_use_norm={"type": "layernorm", "eps": 1e-2}
            ```
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        decoder_interpolation: Interpolation mode used in decoder of the model. Available options are
            **"nearest"**, **"bilinear"**, **"bicubic"**, **"area"**, **"nearest-exact"**. Default is **"nearest"**.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: Unet

    Example:
        .. code-block:: python

            import torch
            import segmentation_models_pytorch as smp

            model = smp.Unet("resnet18", encoder_weights="imagenet", classes=5)
            model.eval()

            # generate random images
            images = torch.rand(2, 3, 256, 256)

            with torch.inference_mode():
                mask = model(images)

            print(mask.shape)
            # torch.Size([2, 5, 256, 256])

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        decoder_layers_configs: Sequence,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "The usage of decoder_use_batchnorm is deprecated. Please modify your code for decoder_use_norm",
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        add_center_block = encoder_name.startswith("vgg")

        self.decoder = UnetDecoderAtt(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            decoder_layers_configs=decoder_layers_configs,
            n_blocks=encoder_depth,
            use_norm=decoder_use_norm,
            add_center_block=add_center_block,
            interpolation_mode=decoder_interpolation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

class UnetDecoderBlockAtt(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.interpolation_mode = config['interpolation_mode']
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        skip_channels = config['skip_channels']

        agg_type = config['aggregation_layer']['layer']
        att1_type = config['attention1']['layer']
        
        # настраиваем cross-attention
        if agg_type == 'conv_cross_att':
            if skip_channels != 0:
                config['aggregation_layer']['params']['q_in_channels'] = in_channels
                config['aggregation_layer']['params']['k_in_channels'] = skip_channels
                config['aggregation_layer']['params']['v_in_channels'] = skip_channels
                config['aggregation_layer']['params']['q_out_channels'] = in_channels
                config['aggregation_layer']['params']['k_out_channels'] = skip_channels
                config['aggregation_layer']['params']['v_out_channels'] = skip_channels

        # настраиваем multihead self-attention, в зависимости от параметров слоя объединения
        if att1_type == 'conv_msa':
            if agg_type == 'conv_cross_att':
                # надо добавить выбор q=features OR q=skip
                if skip_channels != 0:
                    att_channels = in_channels
                else:
                    att_channels = in_channels
                config['attention1']['params']['msa_in_channels'] = att_channels
                config['attention1']['params']['msa_intermediate_channels'] = att_channels
                config['attention1']['params']['msa_out_channels'] = att_channels
                config['attention1']['params']['out_conv_out_channels'] = att_channels
            else:
                config['attention1']['params']['msa_in_channels'] = in_channels + skip_channels
                config['attention1']['params']['msa_intermediate_channels'] = in_channels + skip_channels
                config['attention1']['params']['msa_out_channels'] = in_channels + skip_channels
                config['attention1']['params']['out_conv_out_channels'] = in_channels + skip_channels
        # получаем метод создания слоя агрегации признаков
        create_aggregation = unet_aggregation_factory_dict[agg_type]
        if skip_channels != 0:
            self.aggregation_layer = create_aggregation(**config['aggregation_layer']['params'])
        else:
            self.aggregation_layer = nn.Identity()
        
        # получаем метод создания слоя внимания после агрегации
        create_attention = unet_attention_factory_dict[att1_type]
        self.attention1 = create_attention(**config['attention1']['params'])
        # создаем сверточные слои после слоя внимания
        conv_layers = []
        for idx, params in enumerate(config['conv']):
            if idx == 0:
                if agg_type == 'conv_cross_att' and skip_channels != 0:
                    in_conv_ch = in_channels
                else:
                    in_conv_ch = in_channels + skip_channels
            else:
                in_conv_ch = out_channels

            conv = torchvision.ops.Conv2dNormActivation(
                in_channels=in_conv_ch,
                out_channels=out_channels,
                **params
                )
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)

        att2_type = config['attention2']['layer']
        if att2_type == 'conv_msa':
            config['attention2']['params']['msa_in_channels'] = out_channels
            config['attention2']['params']['msa_intermediate_channels'] = out_channels
            config['attention2']['params']['msa_out_channels'] = out_channels
            config['attention2']['params']['out_conv_out_channels'] = out_channels
        create_attention = unet_attention_factory_dict[att2_type]
        self.attention2 = create_attention(**config['attention2']['params'])

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        #print('DECODER LAYER!!!')
        if skip_connection is not None:
            #print(f'feat:{feature_map.shape},skip:{skip_connection.shape}')
            feature_map = self.aggregation_layer(feature_map, skip_connection)
            feature_map = self.attention1(feature_map)
        #print(f'att_feat:{feature_map.shape}')
        feature_map = self.conv_layers(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map
    
class UnetDecoderAtt(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        decoder_layers_configs: Sequence[Dict],
        n_blocks: int = 5,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
        use_norm:str = "batchnorm",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        
        if decoder_layers_configs is not None and (n_blocks != len(decoder_layers_configs)):
            raise ValueError(
                "Model depth is {}, but you provide `attention_configs` for {} blocks.".format(
                    n_blocks, len(decoder_layers_configs)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = smp.decoders.unet.decoder.UnetCenterBlock(
                head_channels,
                head_channels,
                use_norm=use_norm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels, decoder_layer_config in zip(
            in_channels, skip_channels, out_channels, decoder_layers_configs
        ):
            #print(f'in:{block_in_channels}, skip:{block_skip_channels}, out:{block_out_channels}')
            #print('-------------')
            decoder_layer_config['in_channels'] = block_in_channels
            decoder_layer_config['skip_channels'] = block_skip_channels
            decoder_layer_config['out_channels'] = block_out_channels
            block = UnetDecoderBlockAtt(
                decoder_layer_config
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)

        return x