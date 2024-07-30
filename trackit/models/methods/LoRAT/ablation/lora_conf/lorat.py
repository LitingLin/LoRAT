from typing import Tuple, List, Optional, Mapping, Any
import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from ...modules.patch_embed import PatchEmbedNoSizeCheck
from ...modules.lora import LinearWithLoRA, LinearWithLoRA_TimmQKV
from ...modules.head.mlp import MlpAnchorFreeHead
from typing import List


def find_all_frozen_nn_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and not module.weight.requires_grad:
            lora_module_names.add(name)

    return list(lora_module_names)


def apply_lora(model: nn.Module, lora_module_names: List[str],
               lora_r: int, lora_alpha: float,
               lora_dropout: float = 0.0, use_rslora: bool = False, lora_init_method: str = 'bert',
               lora_target_q: bool = True, lora_target_k: bool = True, lora_target_v: bool = True):
    for lora_module_name in lora_module_names:
        tokens = lora_module_name.split('.')
        parent_module_name = '.'.join(tokens[:-1])
        module_name = tokens[-1]
        parent_module = model.get_submodule(parent_module_name)

        if module_name == 'qkv':
            setattr(parent_module, module_name,
                    LinearWithLoRA_TimmQKV(getattr(parent_module, module_name), lora_r, lora_alpha,
                                           lora_dropout, use_rslora, lora_init_method, lora_target_q, lora_target_k,
                                           lora_target_v))
        else:
            setattr(parent_module, module_name, LinearWithLoRA(getattr(parent_module, module_name), lora_r, lora_alpha,
                                                               lora_dropout, use_rslora, lora_init_method))


class LoRAT_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 lora_r: int, lora_alpha: float, lora_dropout: float, use_rslora: bool,
                 lora_init_method: str,
                 lora_target_q: bool, lora_target_k: bool, lora_target_v: bool, lora_target_o: bool,
                 lora_target_mlp: bool):
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

        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora

        for param in self.parameters():
            param.requires_grad = False

        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            if not lora_target_o:
                linear_names = [name for name in linear_names if not name.endswith('attn.proj')]
            if not lora_target_mlp:
                linear_names = [name for name in linear_names if 'mlp' not in name]
            apply_lora(block, linear_names, lora_r, lora_alpha, lora_dropout, use_rslora,
                       lora_init_method, lora_target_q, lora_target_k, lora_target_v)

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
