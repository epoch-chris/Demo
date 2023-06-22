from typing import Callable
from torch import dtype, nn
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from .vit_attention import ViTSelfAttention
from .vit_mlp import ViTMLP

class ViTBlock(CheckpointModule):
    def __init__(self,

                 hidden_size: int,
                 num_head: int,
                 mlp_ratio: int,
                 activation: Callable,
                 attension_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method:str = 'torch'):
        super().__init__(checkpoint)
        self.norm1 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.attn = ViTSelfAttention(hidden_size=hidden_size,
                                     num_head=num_head,
                                     attension_dropout=attension_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     dtype=dtype,
                                     init_method=init_method)
        self.drop_path = col_nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon,dtype=dtype)
        self.mlp = ViTMLP(hidden_size=hidden_size,
                          mlp_ratio=mlp_ratio,
                          activation=activation,
                          dropout=dropout,
                          dtype=dtype,
                          bias=bias,
                          init_method=init_method)

        def _forward(self, x):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


