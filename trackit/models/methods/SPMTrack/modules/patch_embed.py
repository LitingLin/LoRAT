from typing import Tuple, Optional, Callable
import torch.nn as nn
from torch.nn.modules.utils import _pair
from trackit.models.backbone.dinov2.layers.patch_embed import PatchEmbed


class PatchEmbedNoSizeCheck(nn.Module):
    proj: nn.Conv2d
    norm: nn.LayerNorm
    patch_size: Tuple[int, int]

    def __init__(self,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 patch_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.patch_size = _pair(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    @classmethod
    def build(cls, module: PatchEmbed):
        new_module = cls.__new__(cls)
        nn.Module.__init__(new_module)
        new_module.patch_size = module.proj.kernel_size
        new_module.proj = module.proj
        new_module.norm = module.norm
        return new_module
