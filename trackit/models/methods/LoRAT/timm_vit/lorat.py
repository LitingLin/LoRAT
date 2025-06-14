from typing import Tuple, Optional
import torch
import torch.nn as nn
from timm.layers import trunc_normal_, resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer

from trackit.models import ModelInputDataSelfDescriptionMixin
from .modules.timm_block import TimmViT_CustomBlock
from .modules.timm_dual_stream_to_single_stream import TimmViT_CustomBlock_DualStreamToSingleStream
from ..modules.patch_embed import PatchEmbedNoSizeCheck
from ..modules.head.mlp import MlpAnchorFreeHead
from ..funcs.sample_data import generate_LoRAT_sample_data


class LoRAT_timm_ViT(nn.Module, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: VisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 feature_fusion_end_layer_idx: Optional[int] = None):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, VisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        if feature_fusion_end_layer_idx is None:
            feature_fusion_end_layer_idx = len(vit.blocks)
        self.blocks = nn.ModuleList(TimmViT_CustomBlock.build_from_std_block(block)
                                    if feature_fusion_end_layer_idx != (index + 1)
                                    else TimmViT_CustomBlock_DualStreamToSingleStream.build_from_std_block(block)
                                    for index, block in enumerate(vit.blocks))
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim
        self.pos_drop = vit.pos_drop
        self.norm_pre = vit.norm_pre
        self.grid_size = vit.patch_embed.grid_size

        pos_embed = vit.pos_embed.data[:, vit.num_prefix_tokens if not vit.no_embed_class else 0:, :].clone()
        if self.grid_size[1] < search_region_feat_size[0] or self.grid_size[0] < search_region_feat_size[1]:
            pos_embed = resample_abs_pos_embed(pos_embed, [search_region_feat_size[1], search_region_feat_size[0]],
                                               self.grid_size, 0)
            self.grid_size = (search_region_feat_size[1], search_region_feat_size[0])
        self.pos_embed = nn.Parameter(pos_embed)

        self.feature_fusion_end_layer = feature_fusion_end_layer_idx

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

        z = z + z_pos.to(z.dtype)
        z = self.pos_drop(z)
        z = self.norm_pre(z)
        z = z + self.token_type_embed[z_feat_mask.flatten(1)].to(z.dtype)
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)

        N, L, _ = x.shape
        x_W, x_H = self.x_size
        x_pos = self.pos_embed.view(1, self.grid_size[1], self.grid_size[0], self.embed_dim)
        x_pos = x_pos[:, : x_H, : x_W, :].reshape(1, x_H * x_W, self.embed_dim)
        x = x + x_pos.to(x.dtype)

        x = self.pos_drop(x)
        x = self.norm_pre(x)
        x = x + self.token_type_embed[2].unsqueeze(0).unsqueeze(0).expand(N, L, -1).to(x.dtype)
        return x

    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        L = fusion_feat.shape[1]
        for i in range(self.feature_fusion_end_layer - 1):
            fusion_feat = self.blocks[i](fusion_feat)
        x_feat = self.blocks[self.feature_fusion_end_layer - 1](fusion_feat, L - self.x_size[0] * self.x_size[1])
        for i in range(self.feature_fusion_end_layer, len(self.blocks)):
            x_feat = self.blocks[i](x_feat)
        return self.norm(x_feat)

    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _):
        return generate_LoRAT_sample_data(self.z_size, self.x_size, self.patch_embed.patch_size,
                                          batch_size, device, dtype)
