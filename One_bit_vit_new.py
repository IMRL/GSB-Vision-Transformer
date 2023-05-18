import math
import logging
from functools import partial
from collections import OrderedDict
from weight_init import named_apply, lecun_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers.drop import DropPath
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.weight_init import trunc_normal_
from timm.models.registry import register_model
from quant_new_new import *

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        url='',  # FIXME I have weights for this but > 2GB limit for github release binaries
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9,
        first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', ),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class LearnableBias(nn.Module):
    def __init__(self, out_chn,head):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros([head,1, out_chn//head]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
class LearnableattBias(nn.Module):
    def __init__(self,head, num_patch):
        super(LearnableattBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros([head,num_patch, num_patch]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


class Q_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, config, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        in_features = config.hidden_size
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = QuantizeLinear(in_features, hidden_features, clip_val=config.clip_init_val,
                                  weight_bits=config.weight_bits, input_bits=config.input_bits,
                                  weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                  weight_quant_method=config.weight_quant_method,
                                  input_quant_method=config.input_quant_method,
                                  learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.act = act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = QuantizeLinear(hidden_features, out_features, clip_val=config.clip_init_val,
                                  weight_bits=config.weight_bits, input_bits=config.input_bits,
                                  weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                  weight_quant_method=config.weight_quant_method,
                                  input_quant_method=config.input_quant_method,
                                  learnable=config.learnable_scaling, symmetric=False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        # print(torch.max(x), torch.min(x))
        x = self.act(x)

        x = torch.clip(x, -10., 10.)
        # print(torch.clip(x, -10., 10.))
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Q_Attention(nn.Module):

    def __init__(self, config, quantize_attn=True, qkv_bias=False, attn_drop=0., proj_drop=0.,num_patch=198):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * head_dim

        self.scale = head_dim ** -0.5
        self.quantize_attn = quantize_attn

        self.input_bits = config.input_bits
        # self.norm_q = nn.LayerNorm(head_dim)
        # self.norm_k = nn.LayerNorm(head_dim)
        self.symmetric = config.sym_quant_qkvo
        self.input_layerwise = config.input_layerwise
        self.input_quant_method = config.input_quant_method
        if config.input_quant_method == 'uniform' and config.input_bits < 32:
            self.register_buffer('clip_query', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            
            self.register_buffer('clip_key', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            
            self.register_buffer('clip_value', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_value2', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_value3', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_attn', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_attn2', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_attn3', torch.Tensor([-config.clip_init_val, config.clip_init_val]))

            if config.learnable_scaling:
                self.clip_query = nn.Parameter(self.clip_query)
               
                self.clip_key = nn.Parameter(self.clip_key)
              
                self.clip_value = nn.Parameter(self.clip_value)
                self.clip_value2 = nn.Parameter(self.clip_value2)
                self.clip_value3 = nn.Parameter(self.clip_value3)
                self.clip_attn = nn.Parameter(self.clip_attn)
                self.clip_attn2 = nn.Parameter(self.clip_attn2)
                self.clip_attn3 = nn.Parameter(self.clip_attn3)

        elif (config.input_quant_method == 'elastic' or config.input_quant_method == 'bwn') and config.input_bits < 32:
            self.clip_query = AlphaInit(torch.tensor(1.0))
            
            self.clip_key = AlphaInit(torch.tensor(1.0))
            
            self.clip_value = AlphaInit(torch.tensor(1.0))
            self.clip_value2 = AlphaInit(torch.tensor(1.0))
            self.clip_value3 = AlphaInit(torch.tensor(1.0))
            self.clip_attn = AlphaInit(torch.tensor(1.0))
            self.clip_attn2 = AlphaInit(torch.tensor(1.0))
            self.clip_attn3 = AlphaInit(torch.tensor(1.0))

        if self.quantize_attn:

            self.qkv = QuantizeLinear(config.hidden_size, 3 * config.hidden_size, clip_val=config.clip_init_val,
                                      weight_bits=config.weight_bits, input_bits=config.input_bits,
                                      weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                      weight_quant_method=config.weight_quant_method,
                                      input_quant_method=config.input_quant_method,
                                      learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)

            self.attn_drop = nn.Dropout(attn_drop)

            self.proj = QuantizeLinear(config.hidden_size, config.hidden_size, clip_val=config.clip_init_val,
                                       weight_bits=config.weight_bits, input_bits=config.input_bits,
                                       weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                       weight_quant_method=config.weight_quant_method,
                                       input_quant_method=config.input_quant_method,
                                       learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        else:
            self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        
        self.proj_drop = nn.Dropout(proj_drop)
        self.move_q = LearnableBias(self.all_head_size,self.num_heads)
        self.move_k = LearnableBias(self.all_head_size,self.num_heads)
        self.move_v = LearnableBias(self.all_head_size,self.num_heads)
        self.move_att =LearnableattBias(self.num_heads,num_patch)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # q = self.norm_q(q)
        # k = self.norm_k(k)
        if self.input_bits < 32:
            q = self.move_q(q)
            k = self.move_k(k)
            v = self.move_v(v)


        q = act_quant_fn(q, self.clip_query, num_bits=self.input_bits, symmetric=self.symmetric,
                         quant_method=self.input_quant_method, layerwise=self.input_layerwise)
        k = act_quant_fn(k, self.clip_key, num_bits=self.input_bits, symmetric=self.symmetric,
                         quant_method=self.input_quant_method, layerwise=self.input_layerwise)
        v = qkv_quant_fn(v, self.clip_value,self.clip_value2,self.clip_value3, num_bits=self.input_bits, symmetric=self.symmetric,
                         quant_method='qkv', layerwise=self.input_layerwise)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn= self.move_att(attn)
        attn = self.attn_drop(attn)
        attn = attention_quant_fn(attn, self.clip_attn, self.clip_attn2, self.clip_attn3, num_bits=self.input_bits,
                                  symmetric=False, quant_method='attention', layerwise=self.input_layerwise)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Q_OVitLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
        super(Q_OVitLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Q_Block(nn.Module):

    def __init__(self, config, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=Q_OVitLayerNorm, num_patch=198):
        super().__init__()
        self.norm1 = norm_layer(config.hidden_size)
        self.attn = Q_Attention(config=config, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,num_patch= num_patch)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(config.hidden_size)
        mlp_hidden_dim = int(config.hidden_size * mlp_ratio)
        self.mlp = Q_Mlp(config=config, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Q_PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, nbits, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if nbits ==32:
           self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,padding=0, dilation=1, groups=1, bias=True)
        else:
           self.proj = QuantizeConv2dQ(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, nbits=nbits)
 #          self.proj = Conv2dQ(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, nbits=nbits)
        # nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class lowbit_VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, config, patch_size=16, in_chans=3, num_classes=1000, depth=12,
                 mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=Q_PatchEmbed,
                 norm_layer=Q_OVitLayerNorm, act_layer=gelu, weight_init=''):
        """
        Args:

            patch_size (int, tuple): patch size

            num_classes (int): number of classes for classification head
            config.hidden_size (int): embedding dimension
            depth (int): depth of transformer

            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = config.hidden_size  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer
        act_layer = act_layer

        self.patch_embed = embed_layer(
            nbits=8, img_size=config.input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=config.hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, config.hidden_size))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Q_Block(config, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    num_patch=num_patches + self.num_tokens)
            for i in range(depth)])
        self.norm = norm_layer(self.embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = QuantizeLinear(self.num_features, num_classes, clip_val=config.clip_init_val,
                                   weight_bits=8, input_bits=8,
                                   weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                   weight_quant_method=config.input_quant_method,
                                   input_quant_method=config.input_quant_method,
                                   learnable=config.learnable_scaling,
                                   symmetric=config.sym_quant_qkvo) if num_classes > 0 else nn.Identity()
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            #self.head_dist = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity() 
             self.head_dist = QuantizeLinear(self.num_features, self.num_classes, clip_val=config.clip_init_val,
                                            weight_bits=8, input_bits=8,
                                           weight_layerwise=config.weight_layerwise,
                                           input_layerwise=config.input_layerwise,
                                            weight_quant_method=config.input_quant_method,
                                            input_quant_method=config.input_quant_method,
                                            learnable=config.learnable_scaling,
                                            symmetric=config.sym_quant_qkvo) if num_classes > 0 else nn.Identity()     

        # self.head = LinearQ(self.num_features, num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
        
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = LinearQ(self.embed_dim, self.num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
            
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=''):
    #     _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        # for i,layer in enumerate(list(self.blocks)):
        #     if i%2 ==0:
        #        x = x + layer(x)
        #     else:
        #        x=layer(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def onebits_deit_small_patch16_224(config,num_classes, **kwargs):
    model = lowbit_VisionTransformer(config=config, num_classes=num_classes,patch_size=16, depth=12, mlp_ratio=4, qkv_bias=True,
                                     norm_layer=Q_OVitLayerNorm, **kwargs)
    model.default_cfg = _cfg()

    return model
