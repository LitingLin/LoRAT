from typing import Tuple
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
import math
from torch.nn.modules.utils import _pair
from torch.nn.init import trunc_normal_

from trackit.models import ModelInputDataSelfDescriptionMixin
from trackit.models.backbone.dinov2.model import DinoVisionTransformer, interpolate_pos_encoding
from ...funcs.sample_data import generate_LoRAT_sample_data
from ...modules.patch_embed import PatchEmbedNoSizeCheck
from ...modules.head.mlp import MlpAnchorFreeHead
from ...modules.dinov2_vpt_block import VPTBlock


class LoRAT_DINOv2(nn.Module, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 num_prompt_tokens: int):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = nn.ModuleList([VPTBlock.build(block) for block in vit.blocks])
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        for param in self.parameters():
            param.requires_grad = False

        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

        self.num_prompt_tokens = num_prompt_tokens  # number of prompted tokens

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(self.patch_embed.patch_size), 1) + self.embed_dim))

        num_layers = len(self.blocks)
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_prompt_tokens, self.embed_dim))
            for _ in range(num_layers)
        ])

        for i, prompt_embedding in enumerate(self.prompt_embeddings):
            nn.init.uniform_(prompt_embedding, -val, val)

    def train(self, mode: bool = True):
        super().train(True)
        self.training = mode
        return self

    def forward(self, z: torch.Tensor, x: torch.Tensor, z_feat_mask: torch.Tensor):
        z_feat = self._z_feat(z, z_feat_mask)
        x_feat = self._x_feat(x)
        x_feat = self._fusion(z_feat, x_feat)
        return self.head(x_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        z = z + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        z = z + self.token_type_embed[z_feat_mask.flatten(1)]
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        return x

    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat, self.prompt_embeddings[i])
        fusion_feat = self.norm(fusion_feat)
        return fusion_feat[:, z_feat.shape[1]:, :]

    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _):
        return generate_LoRAT_sample_data(self.z_size, self.x_size, self.patch_embed.patch_size,
                                          batch_size, device, dtype)
