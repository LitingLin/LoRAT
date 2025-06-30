from typing import Tuple, List, Mapping, Any
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from trackit.models.backbone.dinov2.model import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.head.mlp import MlpAnchorFreeHead
from .modules.tmoe.apply import find_all_frozen_nn_linear_names, apply_tmoe
from ... import ModelCacheSelfContainedMixin, ModelInputDataSelfDescriptionMixin


class SPMTrack_Inference_DINOv2(nn.Module, ModelCacheSelfContainedMixin, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 expert_r: int, expert_alpha: float, expert_dropout: float, use_rsexpert: bool = False,
                 expert_nums: int = 4, init_method: str = 'bert', shared_expert: bool = False, route_compression=False):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.track_query = nn.Parameter(torch.empty(1, self.embed_dim))
        self.query_embed = nn.Parameter(torch.empty(1, self.embed_dim))
        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.track_query, std=.02)
        trunc_normal_(self.query_embed, std=.02)
        trunc_normal_(self.token_type_embed, std=.02)

        self.expert_r = expert_r
        self.expert_alpha = expert_alpha
        self.expert_dropout = expert_dropout
        self.use_rsexpert = use_rsexpert
        self.expert_nums = expert_nums
        self.init_method = init_method
        self.shared_expert = shared_expert
        self.route_compression = route_compression

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)
        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block, inference=True)
            apply_tmoe(block, linear_names, self.expert_r, self.expert_alpha, self.expert_dropout, self.use_rsexpert, self.expert_nums, self.init_method, self.shared_expert, self.route_compression)

    def allocate_cache(self, *_):
        self.track_query_dicts = {}

    def destroy_cache(self):
        del self.track_query_dicts

    def forward(self, ids: List[int], z_0: torch.Tensor, x: torch.Tensor, z_0_feat_mask: torch.Tensor,
                         z_1: torch.Tensor = None, z_2: torch.Tensor = None, z_1_feat_mask: torch.Tensor = None, z_2_feat_mask: torch.Tensor = None):
        # 有bug
        ids = torch.as_tensor(ids, dtype=torch.long)
        B = len(ids)
        z_feat = self._z_feat(z_0, z_0_feat_mask)
        z_0_feat = z_feat.clone()
        x = self._x_feat(x)
        if z_1 is not None:
            z_1_feat = self._z_feat(z_1, z_1_feat_mask)
        else:
            z_1_feat = z_0_feat.clone()
        z_feat = torch.cat([z_feat, z_1_feat], dim=1)
        if z_2 is not None:
            z_2_feat = self._z_feat(z_2, z_2_feat_mask)
        else:
            z_2_feat = z_1_feat.clone()
        z_feat = torch.cat([z_feat, z_2_feat], dim=1)

        new_query = []
        for id in ids:
            if self.track_query_dicts.get(id) is not None:
                new_query.append(self.track_query_dicts[id].unsqueeze(0))
            else:
                new_query.append(torch.zeros_like(self.track_query).to(z_feat.device).unsqueeze(0))
        new_query = torch.cat(new_query, dim=0)
        query = new_query + self.track_query.unsqueeze(0).expand(B, 1, -1) + self.query_embed.unsqueeze(0)

        fusion_feat = torch.cat((query, z_feat, x), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)

        enc_opt = fusion_feat[:, -x.size(1):, ...]
        track_query = fusion_feat[:, :1, ...].clone().detach()

        for i, id in enumerate(ids):
            self.track_query_dicts[id] = track_query[i]

        att = torch.matmul(enc_opt, fusion_feat[:, :1, ...].transpose(1, 2))
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 1, 2)).contiguous().squeeze(1)
        output = self.head(opt)
        return output

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

    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _):
        template_feat_map_size = self.z_size
        search_region_feat_map_size = self.x_size
        patch_size = self.patch_embed.patch_size
        template_size = (template_feat_map_size[0] * patch_size[1], template_feat_map_size[1] * patch_size[0])
        search_region_size = (search_region_feat_map_size[0] * patch_size[1], search_region_feat_map_size[1] * patch_size[0])
        return ([i for i in range(batch_size)], # ids
                torch.full((batch_size, 3, template_size[1], template_size[0]), 0.5, device=device), # z_0
                torch.full((batch_size, 3, search_region_size[1], search_region_size[0]), 0.5, device=device), # x
                torch.full((batch_size, template_feat_map_size[1], template_feat_map_size[0]), 1, dtype=torch.long, device=device), # z_0_feat_mask
                torch.full((batch_size, 3, template_size[1], template_size[0]), 0.5, device=device), # z_1
                torch.full((batch_size, 3, template_size[1], template_size[0]), 0.5, device=device), # z_2
                torch.full((batch_size, template_feat_map_size[1], template_feat_map_size[0]), 1, dtype=torch.long, device=device), # z_1_feat_mask
                torch.full((batch_size, template_feat_map_size[1], template_feat_map_size[0]), 1, dtype=torch.long, device=device)) # z_2_feat_mask

    def load_state_dict(
        self, state_dict: Mapping[str, Any], **kwargs
    ):
        if 'expert_alpha' in state_dict: # Remove expert_alpha as it is already specified by the config
            del state_dict['expert_alpha']
        if 'use_rsexpert' in state_dict:  # Remove use_rsexpert as it is already specified by the config
            del state_dict['use_rsexpert']
        return super().load_state_dict(state_dict, **kwargs)
