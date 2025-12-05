import copy
import math
from itertools import chain
from typing import Tuple, Sequence, Optional

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer

from trackit.models import ModelInputDataSelfDescriptionMixin_MultiPath, ModelKVCacheSelfDescriptionMixin
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.block import ChunkedCausalBlock_StaticRoutedLoRAExpert
from .modules.head import MlpAnchorFreeHead


class LoRATv2(nn.Module, ModelInputDataSelfDescriptionMixin_MultiPath, ModelKVCacheSelfDescriptionMixin):
    def __init__(self, vit: VisionTransformer,
                 patch_size: int,
                 template_size: Tuple[int, int],
                 search_region_sizes: Tuple[Tuple[int, int], ...],
                 with_cls_token: bool, with_reg_token: bool,
                 stream_specific_expert_trainable_bits: Tuple[bool, ...],
                 lora_rank: int,
                 lora_alpha: float,
                 lora_dropout: float,
                 enable_flash_attn: bool):
        super().__init__()
        self.template_size = template_size
        self.search_region_sizes = search_region_sizes
        num_search_regions = len(search_region_sizes)

        assert isinstance(vit, VisionTransformer)
        for param in vit.parameters():
            param.requires_grad = False

        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        linear_layer_num_replica = 1 + num_search_regions
        assert len(stream_specific_expert_trainable_bits) == linear_layer_num_replica
        self.blocks = nn.ModuleList(ChunkedCausalBlock_StaticRoutedLoRAExpert.copy_from_std_block(
                                    block, linear_layer_num_replica,
                                    lora_rank, lora_alpha, lora_dropout,
                                    stream_specific_expert_trainable_bits,
                                    enable_flash_attn)
                                    for block in vit.blocks)
        self.embed_dim = vit.embed_dim
        self.pos_drop = vit.pos_drop
        self.norm_pre = vit.norm_pre
        self.grid_size = vit.patch_embed.grid_size
        assert vit.patch_embed.patch_size[0] == patch_size and vit.patch_embed.patch_size[1] == patch_size
        self.patch_size = patch_size

        self.pos_embed = nn.Parameter(torch.empty(1, self.grid_size[0] * self.grid_size[1], self.embed_dim))
        self.pos_embed.data.copy_(vit.pos_embed.data[:, vit.num_prefix_tokens if not vit.no_embed_class else 0:, :])
        self.pos_embed.requires_grad = False

        self.template_token_type_embed = nn.Parameter(torch.empty(2, self.embed_dim))
        nn.init.normal_(self.template_token_type_embed, std=1.e-6)
        if not stream_specific_expert_trainable_bits[0]:
            self.template_token_type_embed.requires_grad = False

        # not necessary, to be removed in LoRATv3
        self.search_region_token_type_embed = nn.ParameterList(tuple(nn.Parameter(torch.empty(1, self.embed_dim)) for _ in range(num_search_regions)))
        for i in range(num_search_regions):
            nn.init.normal_(self.search_region_token_type_embed[i], std=1.e-6)

        self.cls_token = vit.cls_token if with_cls_token else None
        if vit.no_embed_class or not vit.has_class_token or not with_cls_token:
            self.cls_token_pos_embed = None
        else:
            self.cls_token_pos_embed = nn.Parameter(vit.pos_embed[:, 0:1, :])
            self.cls_token_pos_embed.requires_grad = False

        self.num_reg_tokens = vit.num_reg_tokens
        self.reg_token = vit.reg_token if with_reg_token else None
        if vit.no_embed_class or self.num_reg_tokens == 0 or not with_reg_token:
            self.reg_token_pos_embed = None
        else:
            self.reg_token_pos_embed = nn.Parameter(torch.empty(1, self.num_reg_tokens, self.embed_dim))
            self.reg_token_pos_embed.data.copy_(vit.pos_embed[:, int(vit.has_class_token): self.num_reg_tokens + int(vit.has_class_token), :])
            self.reg_token_pos_embed.requires_grad = False

        self.norms = nn.ModuleList(copy.deepcopy(vit.norm) for _ in range(num_search_regions))
        self.heads = nn.ModuleList(MlpAnchorFreeHead(self.embed_dim) for _ in range(num_search_regions))
        for i, (token_type_embed, norm, head, trainable) in enumerate(zip(self.search_region_token_type_embed, self.norms, self.heads, stream_specific_expert_trainable_bits[1:])):
            if not trainable:
                token_type_embed.requires_grad = False
                for param in chain(norm.parameters(), head.parameters()):
                    param.requires_grad = False
        self.enable_flash_attn = enable_flash_attn

    def forward(self, action: str, *args, **kwargs):
        if action == 'train':
            return self._train(*args, **kwargs)
        elif action == 'init':
            return self._init(*args, **kwargs)
        elif action == 'update':
            return self._track(*args, **kwargs, prefill_only=True)
        elif action == 'track':
            return self._track(*args, **kwargs, prefill_only=False)
        else:
            raise ValueError(f'Unknown action: {action}')

    def _init(self, z: torch.Tensor, z_feat_mask: torch.Tensor,
              kv_caches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
              kv_cache_batch_idx: torch.Tensor):
        z_feat = self._z_feat(z, z_feat_mask)
        tokens = self._get_prefix_tokens(z_feat.shape[0])
        tokens.append(z_feat)
        tokens = torch.cat(tokens, dim=1)

        for i in range(len(self.blocks)):
            tokens = self.blocks[i].forward_inference(tokens, 0, kv_caches[i], 0, kv_cache_batch_idx,
                                                      update_kv_cache_only=i == len(self.blocks) - 1)

    def _track(self, index: int, x: torch.Tensor,
                  kv_caches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                  kv_cache_batch_idx: torch.Tensor, prefill_only: bool):
        if isinstance(index, torch.Tensor):
            index = index.item()
        kv_cache_seqlen = sum(self._get_token_chunk_sizes()[:index + 1])

        x_feat = self._x_feat(x, index)
        N, L, _ = x_feat.shape
        for i in range(len(self.blocks)):
            x_feat = self.blocks[i].forward_inference(x_feat, index + 1, kv_caches[i], kv_cache_seqlen, kv_cache_batch_idx,
                                                      update_kv_cache_only=i == len(self.blocks) - 1 and prefill_only)
        if prefill_only:
            return None

        x_feat = self.norms[index](x_feat)
        x_W, x_H = self._get_search_region_feat_size(index)
        return self.heads[-1](x_feat.view(N, x_H, x_W, -1))

    def _train(self, z_list: Tuple[torch.Tensor, ...], x_list: Tuple[torch.Tensor, ...], z_feat_mask: torch.Tensor):
        assert len(z_list) == 1
        z = z_list[0]
        z_feat = self._z_feat(z, z_feat_mask)
        x_feats = tuple(self._x_feat(x, i) for i, x in enumerate(x_list))

        N = z_feat.shape[0]
        tokens = self._get_prefix_tokens(N)
        tokens.append(z_feat)
        tokens.extend(x_feats)
        fusion_feat = torch.cat(tokens, dim=1)
        chunk_sizes = self._get_token_chunk_sizes()
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat, expert_idx=list(range(len(chunk_sizes))), chunk_sizes=chunk_sizes)
        x_feat = fusion_feat[:, sum(chunk_sizes[:-1]):]
        x_feat = self.norms[-1](x_feat)
        x_W, x_H = self._get_search_region_feat_size(-1)
        return self.heads[-1](x_feat.view(N, x_H, x_W, -1))

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self._get_template_feat_size()
        z = z + self.pos_embed.view(1, self.grid_size[0], self.grid_size[1], self.embed_dim)[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        z = self.pos_drop(z)
        z = self.norm_pre(z)
        z = z + self.template_token_type_embed[z_feat_mask.flatten(1)]
        return z

    def _x_feat(self, x: torch.Tensor, index: int):
        N = x.shape[0]
        x = self.patch_embed(x)
        x_W, x_H = self._get_search_region_feat_size(index)
        x = x + self.pos_embed.view(1, self.grid_size[0], self.grid_size[1], self.embed_dim)[:, : x_H, : x_W, :].reshape(1, x_H * x_W, self.embed_dim)
        x = self.pos_drop(x)
        x = self.norm_pre(x)
        x = x + self.search_region_token_type_embed[index].unsqueeze(0).expand(N, x_H * x_W, -1)
        return x

    def get_data_path_names(self, with_train: bool = True, with_eval: bool = True) -> Sequence[str]:
        path_names = []
        if with_train:
            path_names.append('train')
        if with_eval:
            path_names.append('init')
            for i in range(len(self.search_region_sizes)):
                if i != len(self.search_region_sizes) - 1:
                    path_names.append(f'update_{i}')
                else:
                    path_names.append(f'track_{i}')
        return tuple(path_names)

    def get_sample_data(self, name: str, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, auto_mixed_precision_dtype: Optional[torch.dtype]):
        z_size = self.template_size
        z_feat_size = self._get_template_feat_size()
        x_sizes = self.search_region_sizes
        kv_cache_dtype = auto_mixed_precision_dtype if auto_mixed_precision_dtype is not None else dtype
        if name == 'train':
            return ('train', # action
                    [torch.full((batch_size, 3, z_size[1], z_size[0]), 0.5,
                               device=device, dtype=dtype)], # z_list
                    [torch.full((batch_size, 3, x_size[1], x_size[0]), 0.5,
                               device=device, dtype=dtype) for x_size in x_sizes], # x_list
                    torch.full(
                        (batch_size, z_feat_size[1], z_feat_size[0]), 1,
                        dtype=torch.long, device=device) # z_feat_mask
                    )
        elif name == 'init':
            kv_cache_shapes = self.get_kv_cache_shapes(batch_size)
            kv_caches = tuple((torch.full(shape, 0.5, device=device, dtype=kv_cache_dtype),
                              torch.full(shape, 0.5, device=device, dtype=kv_cache_dtype))
                             for shape in kv_cache_shapes)
            kv_cache_batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
            return ('init', # action
                    torch.full((batch_size, 3, z_size[1], z_size[0]), 0.5,
                               device=device, dtype=dtype),  # z
                    torch.full(
                        (batch_size, z_feat_size[1], z_feat_size[0]), 1,
                        dtype=torch.long, device=device),  # z_feat_mask
                    kv_caches,
                    kv_cache_batch_idx
                    )
        else:
            if name.startswith('update_'):
                action = 'update'
                index = int(name.split('_')[1])
            elif name.startswith('track_'):
                action = 'track'
                index = int(name.split('_')[1])
            else:
                raise ValueError(f'Unknown name: {name}')
            kv_cache_shapes = self.get_kv_cache_shapes(batch_size)
            kv_caches = tuple((torch.full(shape, 0.5, device=device, dtype=kv_cache_dtype),
                              torch.full(shape, 0.5, device=device, dtype=kv_cache_dtype))
                             for shape in kv_cache_shapes)
            kv_cache_batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
            return (action,
                    index,
                    torch.full((batch_size, 3, x_sizes[index][1], x_sizes[index][0]), 0.5,
                               device=device, dtype=dtype), # x
                    kv_caches, kv_cache_batch_idx
                    )

    def get_kv_cache_shapes(self, batch_size) -> Sequence[Sequence[int]]:
        depth = len(self.blocks)
        attn = self.blocks[0].attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        num_tokens = sum(self._get_token_chunk_sizes())
        if self.enable_flash_attn:
            return tuple((batch_size, num_tokens, num_heads, head_dim) for _ in range(depth))
        else:
            return tuple((batch_size, num_heads, num_tokens, head_dim) for _ in range(depth))

    def _get_prefix_tokens(self, batch_size: int):
        tokens = []
        num_prefix_tokens = 0
        if self.cls_token is not None:
            tokens.append((self.cls_token + self.cls_token_pos_embed if self.cls_token_pos_embed is not None else self.cls_token).repeat(batch_size, 1, 1))
            num_prefix_tokens += 1
        if self.reg_token is not None:
            tokens.append((self.reg_token + self.reg_token_pos_embed if self.reg_token_pos_embed is not None else self.reg_token).repeat(batch_size, 1, 1))
            num_prefix_tokens += self.num_reg_tokens
        return tokens

    def _get_number_of_prefix_tokens(self):
        num_prefix_tokens = 0
        if self.cls_token is not None:
            num_prefix_tokens += 1
        if self.reg_token is not None:
            num_prefix_tokens += self.num_reg_tokens
        return num_prefix_tokens

    def _get_token_chunk_sizes(self):
        token_chunk_sizes = [self._get_number_of_prefix_tokens() + math.prod(self._get_template_feat_size())]
        for i in range(len(self.search_region_sizes)):
            token_chunk_sizes.append(math.prod(self._get_search_region_feat_size(i)))
        return token_chunk_sizes

    def _get_template_feat_size(self):
        return self.template_size[0] // self.patch_size, self.template_size[1] // self.patch_size

    def _get_search_region_feat_size(self, index: int):
        return self.search_region_sizes[index][0] // self.patch_size, self.search_region_sizes[index][1] // self.patch_size
