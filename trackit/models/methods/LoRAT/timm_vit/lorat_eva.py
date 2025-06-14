from typing import Tuple
import torch
import torch.nn as nn
import torch.amp
from timm.layers import RotaryEmbeddingCat
from timm.layers import trunc_normal_
from timm.models.eva import Eva

from trackit.models import ModelInputDataSelfDescriptionMixin
from ..modules.patch_embed import PatchEmbedNoSizeCheck
from ..modules.eva import CustomEvaAttention
from ..modules.head.mlp import MlpAnchorFreeHead
from ..funcs.sample_data import generate_LoRAT_sample_data


class LoRAT_timm_EVA(nn.Module, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: Eva,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int]):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, Eva)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        for block in vit.blocks:
            block.attn = CustomEvaAttention.build_from_eva_attention(block.attn)
        self.norm = vit.norm

        self.grid_size = vit.patch_embed.grid_size
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.grid_size[0] * self.grid_size[1], self.embed_dim))
        self.pos_embed.data.copy_(vit.pos_embed.data[:, vit.num_prefix_tokens:, :])

        self.pos_drop = vit.pos_drop
        self.embed_dim = vit.embed_dim
        num_heads = vit.blocks[0].attn.num_heads

        assert vit.rope is not None
        self.z_rope = RotaryEmbeddingCat(
            self.embed_dim // num_heads,
            in_pixels=False,
            feat_shape=(self.z_size[1], self.z_size[0]),
            ref_feat_shape=vit.rope.ref_feat_shape,
        )
        self.x_rope = RotaryEmbeddingCat(
            self.embed_dim // num_heads,
            in_pixels=False,
            feat_shape=(self.x_size[1], self.x_size[0]),
            ref_feat_shape=vit.rope.ref_feat_shape,
        )
        assert vit.patch_drop is None

        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z: torch.Tensor, x: torch.Tensor, z_feat_mask: torch.Tensor):
        z_feat = self._z_feat(z, z_feat_mask)
        x_feat = self._x_feat(x)
        x_feat = self._fusion(z_feat, x_feat)
        return self.head(x_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)

        z_W, z_H = self.z_size
        z_pos = self.pos_embed.view(1, self.grid_size[1], self.grid_size[0], self.embed_dim)
        z_pos = z_pos[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)

        z = z + z_pos
        z = self.pos_drop(z)
        z = z + self.token_type_embed[z_feat_mask.flatten(1)]
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)

        N, L, _ = x.shape
        x_W, x_H = self.x_size
        x_pos = self.pos_embed.view(1, self.grid_size[1], self.grid_size[0], self.embed_dim)
        x_pos = x_pos[:, : x_H, : x_W, :].reshape(1, x_H * x_W, self.embed_dim)
        x = x + x_pos

        x = self.pos_drop(x)
        x = x + self.token_type_embed[2].unsqueeze(0).unsqueeze(0).expand(N, L, -1)
        return x

    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        rope = torch.cat((self.z_rope.get_embed(), self.x_rope.get_embed()), dim=0)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat, rope=rope)
        x_feat = fusion_feat[:, z_feat.shape[1]:, :]
        return self.norm(x_feat)

    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _):
        return generate_LoRAT_sample_data(self.z_size, self.x_size, self.patch_embed.patch_size,
                                          batch_size, device, dtype)
