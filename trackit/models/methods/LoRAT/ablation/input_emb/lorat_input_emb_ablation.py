from typing import Tuple, List, Optional, Mapping, Any
import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from ...modules.patch_embed import PatchEmbedNoSizeCheck
from ...modules.lora.apply import find_all_frozen_nn_linear_names, apply_lora
from ...modules.head.mlp import MlpAnchorFreeHead


class LoRAT_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 pos_embed_trainable: bool,
                 enable_token_type_embed: bool, enable_template_foreground_indicating_embed: bool,
                 lora_r: int, lora_alpha: float, lora_dropout: float, use_rslora: bool = False):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        for param in self.parameters():
            param.requires_grad = False

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        if not pos_embed_trainable:
            self.pos_embed.requires_grad = False

        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora

        if enable_token_type_embed:
            if enable_template_foreground_indicating_embed:
                self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
                trunc_normal_(self.token_type_embed, std=.02)
            else:
                self.token_type_embed = nn.Parameter(torch.empty(2, self.embed_dim))
                trunc_normal_(self.token_type_embed, std=.02)

        self.enable_token_type_embed = enable_token_type_embed
        self.enable_template_foreground_indicating_embed = enable_template_foreground_indicating_embed

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_lora(block, linear_names, lora_r, lora_alpha, lora_dropout, use_rslora)

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

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        prefix = kwargs.get('prefix', '')
        for key in list(state_dict.keys()):
            if not self.get_parameter(key[len(prefix):]).requires_grad:
                state_dict.pop(key)
        if self.lora_alpha != 1.:
            state_dict[prefix + 'lora_alpha'] = torch.as_tensor(self.lora_alpha)
            state_dict[prefix + 'use_rslora'] = torch.as_tensor(self.use_rslora)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        if 'lora_alpha' in state_dict:
            state_dict = OrderedDict(**state_dict)
            self.lora_alpha = state_dict['lora_alpha'].item()
            self.use_rslora = state_dict['use_rslora'].item()
            del state_dict['lora_alpha']
            del state_dict['use_rslora']
        return super().load_state_dict(state_dict, **kwargs)
