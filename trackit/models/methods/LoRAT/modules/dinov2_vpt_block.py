import torch
import torch.nn as nn
from torch import Tensor
from trackit.models.backbone.dinov2.layers.block import Block


class VPTBlock(nn.Module):
    @classmethod
    def build(cls, other: Block):
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
        return block

    def forward(self, x: Tensor, vpt_token: Tensor) -> Tensor:
        num_vpt_token = vpt_token.shape[1]
        B = x.shape[0]
        x = torch.cat((vpt_token.expand(B, -1, -1), x), dim=1)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x[:, num_vpt_token:, :]
