from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from trackit.models.backbone.dinov2.model import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.tmoe.apply import find_all_frozen_nn_linear_names, apply_tmoe
from .modules.head.mlp import MlpAnchorFreeHead
from trackit.models import ModelInputDataSelfDescriptionMixin


class SPMTrack_DINOv2(nn.Module, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 expert_r: int, expert_alpha: float, expert_dropout: float, use_rsexpert: bool = False,
                 expert_nums: int = 4, init_method: str = 'bert', shared_expert: bool = False, route_compression: bool = False):
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

        self.expert_alpha = expert_alpha
        self.use_rsexpert = use_rsexpert

        for name, param in self.named_parameters():
            if not ('.experts.' in name or '.gate' in name):
                param.requires_grad = False
        self.track_query = nn.Parameter(torch.empty(1, self.embed_dim))
        self.query_embed = nn.Parameter(torch.empty(1, self.embed_dim))
        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.track_query, std=.02) # not initialized in the original code, but may be better to have it initialized
        trunc_normal_(self.query_embed, std=.02) # not initialized in the original code, but may be better to have it initialized
        trunc_normal_(self.token_type_embed, std=.02)

        # Sometimes the losses do not converge. Please try restarting the experiment with different seed several times

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_tmoe(block, linear_names, expert_r, expert_alpha, expert_dropout, use_rsexpert, expert_nums, init_method, shared_expert, route_compression)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z_0: torch.Tensor, z_1: torch.Tensor, z_2: torch.Tensor,
                x_0: torch.Tensor, x_1: torch.Tensor,
                z_0_feat_mask: torch.Tensor, z_1_feat_mask: torch.Tensor, z_2_feat_mask: torch.Tensor):
        z0_feat = self._z_feat(z_0, z_0_feat_mask)
        z1_feat = self._z_feat(z_1, z_1_feat_mask)
        z2_feat = self._z_feat(z_2, z_2_feat_mask)
        x0_feat = self._x_feat(x_0)
        x1_feat = self._x_feat(x_1)
        return self._multi_frame_predict(z0_feat, z1_feat, z2_feat, x0_feat, x1_feat)

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

    def _multi_frame_predict(self, z_0, z_1, z_2, x_0, x_1):
        z_feat = torch.cat([z_0, z_1, z_2], dim=1)
        B, N, D = z_0.shape

        new_query = self.track_query.unsqueeze(0).expand(B, 1, -1)
        query = new_query + self.query_embed.unsqueeze(0)
        fusion_feat = torch.cat((query, z_feat, x_0), dim=1)

        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        enc_opt = fusion_feat[:, -x_0.size(1):, ...]

        track_query = fusion_feat[:, :1, ...].clone().detach()
        att = torch.matmul(enc_opt, fusion_feat[:, :1, ...].transpose(1, 2))
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 1, 2)).contiguous().squeeze(1)
        output1 = self.head(opt)

        query_2 = track_query + new_query + self.query_embed.unsqueeze(0)
        fusion_feat2 = torch.cat((query_2, z_feat, x_1), dim=1)

        for i in range(len(self.blocks)):
            fusion_feat2 = self.blocks[i](fusion_feat2)
        fusion_feat2 = self.norm(fusion_feat2)
        enc_opt2 = fusion_feat2[:, -x_1.size(1):, ...]

        att2 = torch.matmul(enc_opt2, fusion_feat2[:, :1, ...].transpose(1, 2))
        opt2 = (enc_opt2.unsqueeze(-1) * att2.unsqueeze(-2)).permute((0, 3, 1, 2)).contiguous().squeeze(1)
        output2 = self.head(opt2)

        return output1, output2
    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        return fusion_feat[:, z_feat.shape[1]:, :]

    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _):
        template_feat_map_size = self.z_size
        search_region_feat_map_size = self.x_size
        patch_size = self.patch_embed.patch_size
        template_size = (template_feat_map_size[0] * patch_size[1], template_feat_map_size[1] * patch_size[0])
        search_region_size = (search_region_feat_map_size[0] * patch_size[1], search_region_feat_map_size[1] * patch_size[0])
        return (torch.full((batch_size, 3, template_size[1], template_size[0]), 0.5, device=device),
                torch.full((batch_size, 3, template_size[1], template_size[0]), 0.5, device=device),
                torch.full((batch_size, 3, template_size[1], template_size[0]), 0.5, device=device),
                torch.full((batch_size, 3, search_region_size[1], search_region_size[0]), 0.5, device=device),
                torch.full((batch_size, 3, search_region_size[1], search_region_size[0]), 0.5, device=device),
                torch.full((batch_size, template_feat_map_size[1], template_feat_map_size[0]), 1, dtype=torch.long, device=device),
                torch.full((batch_size, template_feat_map_size[1], template_feat_map_size[0]), 1, dtype=torch.long, device=device),
                torch.full((batch_size, template_feat_map_size[1], template_feat_map_size[0]), 1, dtype=torch.long, device=device))
