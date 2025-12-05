from typing import Tuple, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Attention
from flash_attn import flash_attn_with_kvcache, flash_attn_func

from .lora import build_qkv_linear_layer_replica_with_lora_adapter, build_linear_layer_replica_with_lora_adapter


class ChunkedCausalAttention_StaticRoutedLoRAExpert(nn.Module):
    def __init__(
            self,
            dim: int,
            expert_num_replica: int,
            lora_rank: int,
            lora_alpha: float,
            lora_dropout: float,
            lora_enable_bits: Tuple[bool, ...],
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = build_qkv_linear_layer_replica_with_lora_adapter(nn.Linear(dim, dim * 3, bias=qkv_bias),
                                                                    expert_num_replica,
                                                                    lora_rank, lora_alpha,
                                                                    lora_dropout, lora_enable_bits)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = build_linear_layer_replica_with_lora_adapter(nn.Linear(dim, dim, bias=qkv_bias),
                                                                 expert_num_replica,
                                                                 lora_rank, lora_alpha,
                                                                 lora_dropout, lora_enable_bits)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, expert_indices: Sequence[int] | int, chunk_sizes: Sequence[int]) -> torch.Tensor:
        x_chunks = x.split(chunk_sizes, dim=1)
        last_k = None
        last_v = None

        x_out = []
        for i in range(len(chunk_sizes)):
            x = x_chunks[i]
            B, N, C = x.shape
            expert_index = expert_indices if isinstance(expert_indices, int) else expert_indices[i]
            qkv = self.qkv[expert_index](x).view(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q, k = self.q_norm(q), self.k_norm(k)
            if last_k is not None:
                k = torch.cat([last_k, k], dim=1)
                v = torch.cat([last_v, v], dim=1)

            last_k = k
            last_v = v

            x = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

            x = x.reshape(B, N, C)
            x = self.proj[expert_index](x)
            x = self.proj_drop(x)
            x_out.append(x)
        return torch.cat(x_out, dim=1)

    def forward_inference(self,
                          x: torch.Tensor, expert_index: int,
                          kv_cache: Tuple[torch.Tensor, torch.Tensor],
                          kv_cache_seqlen: int,
                          kv_cache_batch_idx: torch.Tensor,
                          update_kv_cache_only = False):
        B, N, C = x.shape
        if update_kv_cache_only:
            dim = self.qkv[expert_index].weight.shape[1]
            kv_weight = self.qkv[expert_index].weight[dim:]
            kv_bias = self.qkv[expert_index].bias[dim:] if self.qkv[expert_index].bias is not None else None

            kv = F.linear(x, kv_weight, kv_bias).view(B, N, 2, self.num_heads, self.head_dim)
            new_k, new_v = kv.unbind(2)

            new_k = self.k_norm(new_k)
            k_cache, v_cache = kv_cache
            k_cache[kv_cache_batch_idx, kv_cache_seqlen: kv_cache_seqlen + N] = new_k
            v_cache[kv_cache_batch_idx, kv_cache_seqlen: kv_cache_seqlen + N] = new_v
            return None

        qkv = self.qkv[expert_index](x).view(B, N, 3, self.num_heads, self.head_dim)
        q, new_k, new_v = qkv.unbind(2)
        q, new_k = self.q_norm(q), self.k_norm(new_k)
        k_cache, v_cache = kv_cache

        kv_cache_seqlens = torch.full((kv_cache_batch_idx.shape[0],), kv_cache_seqlen, dtype=torch.int32, device=kv_cache_batch_idx.device)
        x = flash_attn_with_kvcache(q, k_cache, v_cache, new_k, new_v,
                                    cache_seqlens=kv_cache_seqlens,
                                    cache_batch_idx=kv_cache_batch_idx)

        x = x.reshape(B, N, C)
        x = self.proj[expert_index](x)
        x = self.proj_drop(x)
        return x


    @staticmethod
    def copy_from_std_attn(other: Attention,
                           expert_num_replica: int,
                           lora_rank: int,
                           lora_alpha: float,
                           lora_dropout: float,
                           lora_enable_bits: Tuple[bool, ...]):
        attn = ChunkedCausalAttention_StaticRoutedLoRAExpert.__new__(ChunkedCausalAttention_StaticRoutedLoRAExpert)
        nn.Module.__init__(attn)
        attn.num_heads = other.num_heads
        attn.head_dim = other.head_dim
        attn.qkv = build_qkv_linear_layer_replica_with_lora_adapter(other.qkv,
                                                                    expert_num_replica,
                                                                    lora_rank,
                                                                    lora_alpha,
                                                                    lora_dropout,
                                                                    lora_enable_bits)
        attn.q_norm = other.q_norm
        attn.k_norm = other.k_norm
        attn.proj = build_linear_layer_replica_with_lora_adapter(other.proj,
                                                                 expert_num_replica,
                                                                 lora_rank,
                                                                 lora_alpha,
                                                                 lora_dropout,
                                                                 lora_enable_bits)
        attn.fused_attn = other.fused_attn
        attn.attn_drop = other.attn_drop
        attn.proj_drop = other.proj_drop
        attn.scale = other.scale
        return attn
