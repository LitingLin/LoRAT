from typing import Tuple, Mapping, Any
import torch
import torch.nn as nn
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from ...modules.patch_embed import PatchEmbedNoSizeCheck
from ...modules.head.mlp import MlpAnchorFreeHead
from ...modules.lora.merge import lora_merge_state_dict


class LoRATBaseline_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int]):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.z_pos_embed = nn.Parameter(torch.empty(1, template_feat_size[0] * template_feat_size[1], self.embed_dim))
        self.z_pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                             self.z_size,
                                                             vit.patch_embed.patches_resolution,
                                                             num_prefix_tokens=0, interpolate_offset=0))

        self.x_pos_embed = nn.Parameter(torch.empty(1, search_region_feat_size[0] * search_region_feat_size[1], self.embed_dim))
        self.x_pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                             self.x_size,
                                                             vit.patch_embed.patches_resolution,
                                                             num_prefix_tokens=0, interpolate_offset=0))

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z: torch.Tensor, x: torch.Tensor, z_feat_mask: torch.Tensor):
        z_feat = self._z_feat(z, z_feat_mask)
        x_feat = self._x_feat(x)
        x_feat = self._fusion(z_feat, x_feat)
        return self.head(x_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z = z + self.z_pos_embed
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.x_pos_embed
        return x

    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        return fusion_feat[:, z_feat.shape[1]:, :]

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        state_dict = lora_merge_state_dict(self, state_dict)
        return super().load_state_dict(state_dict, **kwargs)
