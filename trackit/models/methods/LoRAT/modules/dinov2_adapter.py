# https://github.com/google-research/adapter-bert/blob/master/modeling.py
# https://github.com/KMnP/vpt/blob/main/src/models/vit_adapter/adapter_block.py

import torch
import torch.nn as nn
from torch import Tensor
from trackit.models.backbone.dinov2 import Block


class DINOv2AdapterBlock(nn.Module):
    @classmethod
    def build(cls, other: Block, reduction_factor: int):
        block = cls.__new__(cls)
        nn.Module.__init__(block)
        block.norm1 = other.norm1
        block.attn = other.attn
        block.ls1 = other.ls1
        block.drop_path1 = other.drop_path1
        block.norm2 = other.norm2
        block.mlp = other.mlp
        block.ls2 = other.ls2
        block.drop_path2 = other.drop_path2
        block.sample_drop_ratio = other.sample_drop_ratio

        hidden_dim = other.attn.proj.weight.shape[0]
        block.adapter_downsample = nn.Linear(
            hidden_dim,
            hidden_dim // reduction_factor
        )
        block.adapter_upsample = nn.Linear(
            hidden_dim // reduction_factor,
            hidden_dim
        )
        block.adapter_act_fn = nn.GELU()

        nn.init.zeros_(block.adapter_downsample.weight)
        nn.init.zeros_(block.adapter_downsample.bias)

        nn.init.zeros_(block.adapter_upsample.weight)
        nn.init.zeros_(block.adapter_upsample.bias)

        return block

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        adpt = self.adapter_downsample(x_)
        adpt = self.adapter_act_fn(adpt)
        adpt = self.adapter_upsample(adpt)
        x = x + adpt + x_

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
