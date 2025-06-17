from itertools import chain
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from trackit.models import ModelInputDataSelfDescriptionMixin
from trackit.models.backbone.dinov2.model import DinoVisionTransformer, interpolate_pos_encoding
from ...funcs.sample_data import generate_LoRAT_sample_data
from ...modules.patch_embed import PatchEmbedNoSizeCheck
from ...modules.head.mlp import MlpAnchorFreeHead


class LoRAT_DINOv2(nn.Module, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 enable_token_type_embed: bool, enable_template_foreground_indicating_embed: bool):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        if enable_token_type_embed:
            if enable_template_foreground_indicating_embed:
                self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
                trunc_normal_(self.token_type_embed, std=.02)
            else:
                self.token_type_embed = nn.Parameter(torch.empty(2, self.embed_dim))
                trunc_normal_(self.token_type_embed, std=.02)

        self.enable_token_type_embed = enable_token_type_embed
        self.enable_template_foreground_indicating_embed = enable_template_foreground_indicating_embed

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z: torch.Tensor, x: torch.Tensor, z_feat_mask: torch.Tensor):
        z_feat = self._z_feat(z, z_feat_mask)
        x_feat = self._x_feat(x)
        x_feat = self._fusion(z_feat, x_feat)
        return self.head(x_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        z = z + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        if self.enable_token_type_embed:
            if self.enable_template_foreground_indicating_embed:
                z = z + self.token_type_embed[z_feat_mask.flatten(1)]
            else:
                z = z + self.token_type_embed[0].view(1, 1, self.embed_dim)
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        if self.enable_token_type_embed:
            if self.enable_template_foreground_indicating_embed:
                x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
            else:
                x = x + self.token_type_embed[1].view(1, 1, self.embed_dim)
        return x

    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        return fusion_feat[:, z_feat.shape[1]:, :]

    def freeze_for_peft(self, pos_embed_trainable: bool):
        to_freeze = [self.patch_embed.parameters(),
            self.blocks.parameters(),
            (self.norm,)]
        if not pos_embed_trainable:
            to_freeze.append((self.pos_embed,))
        for p in chain(to_freeze):
            p.requires_grad = False

    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _):
        return generate_LoRAT_sample_data(self.z_size, self.x_size, self.patch_embed.patch_size,
                                          batch_size, device, dtype)
