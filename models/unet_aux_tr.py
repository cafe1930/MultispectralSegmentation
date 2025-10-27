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
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from typing import Any, Dict, Optional, Union, Callable, Sequence, List
import einops as eo

from functools import partial
from torchvision.models.vision_transformer import EncoderBlock as VitEncoderBlock, MLPBlock


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            _freeze=False,
            device=None,
            dtype=None
            ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype
            )
    def forward(self, x):
        #!!!!
        bs, seq_len, emb_dim = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long).to(x.device)
        pos_embeddings = self.embedding(positions)
        return x + pos_embeddings
    
class FixedSizeLearnableEmbeddings(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            ):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.empty(1, num_embeddings, embedding_dim).normal_(std=0.02))

    def forward(self,x):
        return x + self.positional_encoding

class ComputeWeights(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        mod_lst = []
        mod_lst.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
        mod_lst.append(nn.Sigmoid())
        super().__init__(*mod_lst)
        

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.in2_proj = ComputeWeights(in_channels, out_channels, kernel_size, padding)

    def forward(self, in1, in2):
        #print(in1.shape, in2.shape)
        weights = self.in2_proj(in2)
        bs, ch1, rows1, cols1 = in1.shape
        bs, ch2, rows2, cols2 = weights.shape
        winr = rows1//rows2
        winc = cols1//cols2

        weights = weights.view(bs, ch2, rows2, cols2, 1, 1)

        in1 = eo.rearrange(in1, 'b ch (rn wr) (cn wc) -> b ch rn cn wr wc', rn=rows2, cn=cols2, wr=winr, wc=winc)

        in1 = in1*weights

        in1 = eo.rearrange(in1, 'b ch rn cn wr wc -> b ch (rn wr) (cn wc)', rn=rows2, cn=cols2, wr=winr, wc=winc)
        return in1

class VisionTransformerBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int, # is equal to query_dim
            kdim: int,
            vdim: int,
            mlp_dim: int,
            attention_layer: Callable[..., torch.nn.Module],
            dropout:float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 ):
        super().__init__()
        assert kdim==vdim, f"kdim should be equal to vdim! kdim={kdim}, vdim={vdim}"
        
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.hidden_dim = hidden_dim

        # Attention block
        self.ln_11 = norm_layer(hidden_dim)
        self.ln_12 = norm_layer(kdim)

        if attention_layer is nn.MultiheadAttention:
            
            attention_layer = partial(attention_layer, batch_first=True)
        self.self_attention = attention_layer(hidden_dim, num_heads, kdim=kdim, vdim=vdim,)
        #if kdim==vdim==hidden_dim:    
        #else:self.self_attention = attention_layer(hidden_dim, num_heads,)
        
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, target, source=None, need_weghts=False):
        x = self.ln_11(target)
        
        if source is None:
            # Self Attention
            x, weights = self.self_attention(query=x, key=x, value=x, average_attn_weights=False, need_weights=need_weghts)

        else:
            y = self.ln_12(source)

            # Cross Attention
            x, weights = self.self_attention(query=x, key=y, value=y, average_attn_weights=False, need_weights=need_weghts)

        x = self.dropout(x)
        x = x + target

        x2 = self.ln_2(x)
        x2 = self.mlp(x2)
        return x2 + x, weights

class WindowVisionTransformer(nn.Module):
    def __init__(
            self,
            cols_in_patch:int,
            rows_in_patch:int,
            channels:int,
            # transformers block params
            num_heads:int,
            mlp_dim:int,
            dropout:float,
            layer_num:int,
            transformer_type:'str', # channels and patches are possible
            positional_encoding: Callable[..., nn.Module] = nn.Identity,
            ):
    
        super().__init__()

        self.cols_in_patch = cols_in_patch
        self.rows_in_patch = rows_in_patch
        self.transformer_type =  transformer_type
        
        if transformer_type == 'channels':
            self.seq_len = channels
            hidden_dim = cols_in_patch * rows_in_patch
            
        elif transformer_type == 'patches':
            self.seq_len = cols_in_patch * rows_in_patch
            hidden_dim = channels


        
        self.positional_encoding = positional_encoding(num_embeddings=self.seq_len, embedding_dim=hidden_dim)
        # можно создать несколько трансформерных слоев
        transformer_layers_list = [
            VisionTransformerBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                kdim=hidden_dim,
                vdim=hidden_dim,
                mlp_dim=mlp_dim,
                attention_layer=nn.MultiheadAttention,
                dropout=dropout)
            for i in range(layer_num)
        ]
        self.transformer_layers = nn.ModuleList(transformer_layers_list)
        
    def forward(self, x):
        bs, channels, rows, cols = x.shape

        row_patch_num = rows//self.rows_in_patch
        col_patch_num = cols//self.cols_in_patch
        # размер (bs, channels, rows, cols) преобразовываем в размер (row_patches*col_patches, bs, channels, rows_in_patch*cols_in_patch)
        # rows=row_patches*rows_in_patch, cols=col_patches*cols_in_patch
        if self.transformer_type == 'channels':
            rearrange_pattern = 'bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch) -> (row_patch_num col_patch_num) bs channels (rows_in_patch cols_in_patch)'
            rearrange_args = {
                'cols_in_patch':self.cols_in_patch,
                'rows_in_patch':self.rows_in_patch,
            }
            
        elif self.transformer_type == 'patches':
            rearrange_pattern = 'bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch) -> (row_patch_num col_patch_num) bs (rows_in_patch cols_in_patch) channels'
            rearrange_args = {
                'cols_in_patch':self.cols_in_patch,
                'rows_in_patch':self.rows_in_patch,
            }
        
        h = eo.rearrange(
            x,
            rearrange_pattern,
            **rearrange_args,
        )
        # позиционное кодирование (его может и не быть - nn.Identity)
        layer_outs = self.positional_encoding(h)
        processed_outs = []

        # итерирование по окнам 
        for i, layer_out in enumerate(layer_outs):

            for layer in self.transformer_layers:

                layer_out, layer_att_weights = layer(layer_out)
            processed_outs.append(layer_out.unsqueeze(0))
        
        processed_outs = torch.cat(processed_outs,dim=0)
        if self.transformer_type == 'channels':
            rearrange_pattern = '(row_patch_num col_patch_num) bs channels (rows_in_patch cols_in_patch) -> bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch)'
            rearrange_args = {
                'cols_in_patch':self.cols_in_patch,
                'rows_in_patch':self.rows_in_patch,
                'row_patch_num':row_patch_num,
                'col_patch_num':col_patch_num,
            }
        elif self.transformer_type == 'patches':
            rearrange_pattern = '(row_patch_num col_patch_num) bs (rows_in_patch cols_in_patch) channels -> bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch)'
            rearrange_args = {
                'cols_in_patch':self.cols_in_patch,
                'rows_in_patch':self.rows_in_patch,
                'row_patch_num':row_patch_num,
                'col_patch_num':col_patch_num,
            }
        processed_outs = eo.rearrange(
            processed_outs,
            rearrange_pattern,
            **rearrange_args,
        )
        return processed_outs


class WindowCrossAttention(nn.Module):
    def __init__(
            self,
            cols_in_patch_x:int,
            rows_in_patch_x:int,
            cols_in_patch_y:int,
            rows_in_patch_y:int,
            channels_x:int,
            channels_y:int,
            # transformers block params
            num_heads:int,
            mlp_dim:int,
            dropout:float,
            transformer_type:'str', # channels and patches are possible
            positional_encoding_x: Callable[..., nn.Module] = nn.Identity,
            positional_encoding_y: Callable[..., nn.Module] = nn.Identity,
            ):
    
        super().__init__()
        self.transformer_type =  transformer_type
        self.cols_in_patch_x = cols_in_patch_x
        self.rows_in_patch_x = rows_in_patch_x

        self.cols_in_patch_y = cols_in_patch_y
        self.rows_in_patch_y = rows_in_patch_y
        
        
        if transformer_type == 'channels':
            self.seq_len_x = channels_x
            hidden_dim_x = cols_in_patch_x * rows_in_patch_x
            self.seq_len_y = channels_y
            hidden_dim_y = cols_in_patch_y * rows_in_patch_y
            
        elif transformer_type == 'patches':
            self.seq_len_x = cols_in_patch_x * rows_in_patch_x
            hidden_dim_x = channels_x
            self.seq_len_y = cols_in_patch_y * rows_in_patch_y
            hidden_dim_y = channels_y

        
        self.positional_encoding_x = positional_encoding_x(num_embeddings=self.seq_len_x, embedding_dim=hidden_dim_x)
        self.positional_encoding_y = positional_encoding_y(num_embeddings=self.seq_len_y, embedding_dim=hidden_dim_y)
        # можно создать несколько трансформерных слоев
        self.cross_attention_block = VisionTransformerBlock(
            num_heads=num_heads,
            hidden_dim=hidden_dim_x,
            kdim=hidden_dim_y,
            vdim=hidden_dim_y,
            mlp_dim=mlp_dim,
            attention_layer=nn.MultiheadAttention,
            dropout=dropout)
        
    def forward(self, x, y):
        bs, channels, rows_x, cols_x = x.shape
        
        row_patch_num_x = rows_x//self.rows_in_patch_x
        col_patch_num_x = cols_x//self.cols_in_patch_x

        bs, channels, rows_y, cols_y = y.shape
        row_patch_num_y = rows_y//self.rows_in_patch_y
        col_patch_num_y = cols_y//self.cols_in_patch_y

        

        assert row_patch_num_x == row_patch_num_y, f'number of window rows in X={row_patch_num_x} and Y={row_patch_num_y} tensors ought to coinside'
        assert col_patch_num_x == col_patch_num_y, f'number of window cols in X={col_patch_num_x} and Y={col_patch_num_y} tensors ought to coinside'

        # размер (bs, channels, rows, cols) преобразовываем в размер (row_patches*col_patches, bs, channels, rows_in_patch*cols_in_patch)
        # rows=row_patches*rows_in_patch, cols=col_patches*cols_in_patch
        if self.transformer_type == 'channels':
            rearrange_pattern = 'bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch) -> (row_patch_num col_patch_num) bs channels (rows_in_patch cols_in_patch)'
            rearrange_args_x = {
                'cols_in_patch':self.cols_in_patch_x,
                'rows_in_patch':self.rows_in_patch_x,
            }

            rearrange_args_y = {
                'cols_in_patch':self.cols_in_patch_y,
                'rows_in_patch':self.rows_in_patch_y,
            }
             
        elif self.transformer_type == 'patches':
            rearrange_pattern = 'bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch) -> (row_patch_num col_patch_num) bs (rows_in_patch cols_in_patch) channels'
            rearrange_args_x = {
                'cols_in_patch':self.cols_in_patch_x,
                'rows_in_patch':self.rows_in_patch_x,
            }
            rearrange_args_y = {
                'cols_in_patch':self.cols_in_patch_y,
                'rows_in_patch':self.rows_in_patch_y,
            }
        
        hx = eo.rearrange(
            x,
            rearrange_pattern,
            **rearrange_args_x,
        )
        hy = eo.rearrange(
            y,
            rearrange_pattern,
            **rearrange_args_y,
        )

        # позиционное кодирование (его может и не быть - nn.Identity)
        
        layer_outs_x = self.positional_encoding_x(hx)
        layer_outs_y = self.positional_encoding_y(hy)
        processed_outs = []

        # итерирование по окнам 
        for i, (layer_out_x, layer_out_y) in enumerate(zip(layer_outs_x, layer_outs_y)):
            layer_out, layer_att_weights = self.cross_attention_block(layer_out_x, layer_out_y)
            processed_outs.append(layer_out.unsqueeze(0))
        
        processed_outs = torch.cat(processed_outs,dim=0)
        if self.transformer_type == 'channels':
            rearrange_pattern = '(row_patch_num col_patch_num) bs channels (rows_in_patch cols_in_patch) -> bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch)'
            rearrange_args = {
                'cols_in_patch':self.cols_in_patch_x,
                'rows_in_patch':self.rows_in_patch_x,
                'row_patch_num':row_patch_num_x,
                'col_patch_num':col_patch_num_x,
            }
        elif self.transformer_type == 'patches':
            rearrange_pattern = '(row_patch_num col_patch_num) bs (rows_in_patch cols_in_patch) channels -> bs channels (row_patch_num rows_in_patch) (col_patch_num cols_in_patch)'
            rearrange_args = {
                'cols_in_patch':self.cols_in_patch_x,
                'rows_in_patch':self.rows_in_patch_x,
                'row_patch_num':row_patch_num_x,
                'col_patch_num':col_patch_num_x,
            }
        processed_outs = eo.rearrange(
            processed_outs,
            rearrange_pattern,
            **rearrange_args,
        )
        return processed_outs
    
class HyperspectralTransformer(nn.Module):
    def __init__(
            self,
            config
            ):
    
        super().__init__()
        self.config = config
        self.patch_emd = config['patch_emd']['layer'](**config['patch_emd']['params'])
        transformer_layers = {}
        for i, transformer_layer_config in enumerate(config['transformer_layers']):

            if transformer_layer_config['layer'] == 'crossatt':
                transformer_layer_config['params']['positional_encoding_x'] = pos_enc_factory_dict[transformer_layer_config['params']['positional_encoding_x']]
                transformer_layer_config['params']['positional_encoding_y'] = pos_enc_factory_dict[transformer_layer_config['params']['positional_encoding_y']]
            else:
                transformer_layer_config['params']['positional_encoding'] = pos_enc_factory_dict[transformer_layer_config['params']['positional_encoding']]
            layer_name = transformer_layer_config['layer']
            layer_creat = transformer_factory_dict[layer_name]
            transformer_layers[f'{i}_{layer_name}'] = layer_creat(**transformer_layer_config['params'])

        self.transformer_layers = nn.ModuleDict(transformer_layers)
        #self.output_layer = config['output_layer']['layer'](**config['patch_emd']['params'])

    def forward(self, x, y: Dict[str, torch.Tensor]):
        x = self.patch_emd(x)
        results = []
        for i, (name, layer) in enumerate(self.transformer_layers.items()):
            if name in y:       
                x = layer(x, y[name])
            else:
                x = layer(x)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            _freeze=False,
            device=None,
            dtype=None
            ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype
            )
    def forward(self, x):
        #!!!!
        bs, seq_len, emb_dim = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long).to(x.device)
        pos_embeddings = self.embedding(positions)
        return x + pos_embeddings
    
class FixedSizeLearnableEmbeddings(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            ):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.empty(1, num_embeddings, embedding_dim).normal_(std=0.02))

    def forward(self,x):
        return x + self.positional_encoding

class AddFeatures(nn.Module):
    def forward(self, x, y):
        return x+y

class ConcatFeatures(nn.Module):
    '''
    Implementation of concatenation. It is nececcary for various aggreagation strategies in UNet decoder
    '''
    def forward(self, *tensors, dim=1):
        return torch.cat(tensors, dim=dim)
    
class PassOneFeature(nn.Module):
    def __init__(self, passing_idx):
        super().__init__()
        self.passing_idx = passing_idx
    def forward(self, *tensors):
        return tensors[self.passing_idx]

    

pos_enc_factory_dict = {
    'fixed_embeddings': FixedSizeLearnableEmbeddings,
    'embedding_layer': EmbeddingLayer,
    'none': nn.Identity
}

transformer_factory_dict = {
    'win_mha': WindowVisionTransformer,
    'crossatt': WindowCrossAttention,
    'none': nn.Identity,
}

feature_aggregation_factory_dict ={
    'add': AddFeatures,
    'pass_one': PassOneFeature,
    'concat': ConcatFeatures,
    'crossatt': WindowCrossAttention,
    'channel_att': ChannelAtt,
}

class UnetAuxAtt(SegmentationModel):
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
        aux_transformer_config: Dict[str, Any],
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

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
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
        #print(aux_transformer_config['input_transformer']['params'])
        aux_transformer_config['output_crossatt']['params']['positional_encoding_x'] = pos_enc_factory_dict[aux_transformer_config['output_crossatt']['params']['positional_encoding_x']]
        aux_transformer_config['output_crossatt']['params']['positional_encoding_y'] = pos_enc_factory_dict[aux_transformer_config['output_crossatt']['params']['positional_encoding_y']]
        aux_transformer_config['input_transformer']['params']['positional_encoding'] = pos_enc_factory_dict[aux_transformer_config['input_transformer']['params']['positional_encoding']]
        aux_transformer_config['intermediate_layers']['params']['positional_encoding'] = pos_enc_factory_dict[aux_transformer_config['intermediate_layers']['params']['positional_encoding']]
        #config['hsi_augmentation']['layer'] = feature_aggregation_factory_dict['hsi_augmentation']['layer']
        if aux_transformer_config['hsi_augmentation']['layer'] == 'crossatt':
            aux_transformer_config['hsi_augmentation']['params']['positional_encoding_x'] = pos_enc_factory_dict[aux_transformer_config['hsi_augmentation']['params']['positional_encoding_x']]
            aux_transformer_config['hsi_augmentation']['params']['positional_encoding_y'] = pos_enc_factory_dict[aux_transformer_config['hsi_augmentation']['params']['positional_encoding_y']]

        self.aux_transf = nn.ModuleDict()
        self.aux_transf['patch_emd'] = aux_transformer_config['patch_emd']['layer'](**aux_transformer_config['patch_emd']['params'])
        layer_name = aux_transformer_config['input_transformer']['layer']
        #print(aux_transformer_config['input_transformer']['params'])
        #print()
        self.aux_transf['input_transformer'] = transformer_factory_dict[layer_name](**aux_transformer_config['input_transformer']['params'])
        self.aux_transf['hsi_augmentation'] = feature_aggregation_factory_dict[aux_transformer_config['hsi_augmentation']['layer']](**aux_transformer_config['hsi_augmentation']['params'])
        layer_name = aux_transformer_config['intermediate_layers']['layer']
        self.aux_transf['intermediate_layers'] = transformer_factory_dict[layer_name](**aux_transformer_config['intermediate_layers']['params'])
        layer_name = aux_transformer_config['output_crossatt']['layer']
        self.aux_transf['output_crossatt'] = transformer_factory_dict[layer_name](**aux_transformer_config['output_crossatt']['params'])
        self.aux_transf['output_layer'] = aux_transformer_config['output_layer']['layer'](**aux_transformer_config['output_layer']['params'])

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        aug_x = self.aux_transf['patch_emd'](x)
        
        aug_x = self.aux_transf['input_transformer'](aug_x)

        x = self.aux_transf['hsi_augmentation'](x, aug_x)

        features = self.encoder(x)
        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        aug_x = self.aux_transf['intermediate_layers'](aug_x)
        masks = self.aux_transf['output_crossatt'](masks, aug_x)
        masks = self.aux_transf['output_layer'](masks)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks