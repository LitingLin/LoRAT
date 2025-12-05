from typing import Callable, Optional, Tuple, Sequence

import torch
from torch import nn, Tensor

from timm.models.vision_transformer import LayerScale, DropPath, Block
from .mlp import Mlp_StaticRoutedLoRAExpert


class ChunkedCausalBlock_StaticRoutedLoRAExpert(nn.Module):
    def __init__(
            self,
            dim: int,
            expert_num_replica: int,
            lora_rank: int,
            lora_alpha: float,
            lora_dropout: float,
            lora_enable_bits: Tuple[bool, ...],
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            enable_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        if enable_flash_attn:
            from .attention_flash_attn import ChunkedCausalAttention_StaticRoutedLoRAExpert
        else:
            from .attention import ChunkedCausalAttention_StaticRoutedLoRAExpert
        self.norm1 = norm_layer(dim)
        self.attn = ChunkedCausalAttention_StaticRoutedLoRAExpert(
            dim,
            expert_num_replica,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_enable_bits,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_StaticRoutedLoRAExpert(
            expert_num_replica,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_enable_bits,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, expert_idx: Sequence[int] | int, chunk_sizes: None | Sequence[int]) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), expert_idx, chunk_sizes))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x), expert_idx, chunk_sizes))

        if self.training and isinstance(self.drop_path1, DropPath) and self.drop_path1.drop_prob > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.drop_path1.drop_prob,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.drop_path2.drop_prob,
            )
        elif self.training and isinstance(self.drop_path1, DropPath):
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)

        return x

    def forward_inference(self,
                          x: torch.Tensor, expert_index: int,
                          kv_cache: Tuple[torch.Tensor, torch.Tensor],
                          kv_cache_seqlen: int,
                          kv_cache_batch_idx: torch.Tensor,
                          update_kv_cache_only: bool):
        h = self.attn.forward_inference(self.norm1(x), expert_index, kv_cache, kv_cache_seqlen,
                                        kv_cache_batch_idx, update_kv_cache_only)
        if update_kv_cache_only:
            return None
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), expert_index)))
        return x

    @staticmethod
    def copy_from_std_block(other: Block,
                            expert_num_replica: int,
                            lora_rank: int,
                            lora_alpha: float,
                            lora_dropout: float,
                            lora_enable_bits: Tuple[bool, ...],
                            enable_flash_attn: bool):
        block = ChunkedCausalBlock_StaticRoutedLoRAExpert.__new__(ChunkedCausalBlock_StaticRoutedLoRAExpert)
        nn.Module.__init__(block)
        if enable_flash_attn:
            from .attention_flash_attn import ChunkedCausalAttention_StaticRoutedLoRAExpert
        else:
            from .attention import ChunkedCausalAttention_StaticRoutedLoRAExpert
        block.norm1 = other.norm1
        block.attn = ChunkedCausalAttention_StaticRoutedLoRAExpert.copy_from_std_attn(
            other.attn, expert_num_replica, lora_rank, lora_alpha, lora_dropout, lora_enable_bits)
        block.ls1 = other.ls1
        block.drop_path1 = other.drop_path1
        block.norm2 = other.norm2
        block.mlp = Mlp_StaticRoutedLoRAExpert.copy_from_std_block(
            other.mlp, expert_num_replica, lora_rank, lora_alpha, lora_dropout, lora_enable_bits)
        block.ls2 = other.ls2
        block.drop_path2 = other.drop_path2
        return block


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)
