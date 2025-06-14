from typing import Callable, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, Attention, LayerScale, DropPath, use_fused_attn, Block

from .timm_block import drop_add_residual_stochastic_depth


class TimmViT_CustomBlock_DualStreamToSingleStream(nn.Module):
    def __init__(
            self,
            dim: int,
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
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TimmViT_CustomAttention_DualStreamToSingleStream(
            dim,
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
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor, tokens_to_drop: int) -> Tensor:
        def attn_residual_func(x: Tensor, tokens_to_drop: int) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), tokens_to_drop))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and isinstance(self.drop_path1, DropPath) and self.drop_path1.drop_prob > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth_dual_stream_to_single_stream(
                x, tokens_to_drop,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.drop_path1.drop_prob,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.drop_path2.drop_prob,
            )
        elif self.training and isinstance(self.drop_path1, DropPath):
            h = self.drop_path1(attn_residual_func(x, tokens_to_drop))
            x = x.narrow(1, tokens_to_drop, x.size(1) - tokens_to_drop)
            x = x + h
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            h = attn_residual_func(x, tokens_to_drop)
            x = x.narrow(1, tokens_to_drop, x.size(1) - tokens_to_drop)
            x = x + h
            x = x + ffn_residual_func(x)
        return x


    @staticmethod
    def build_from_std_block(other: Block):
        block = TimmViT_CustomBlock_DualStreamToSingleStream.__new__(TimmViT_CustomBlock_DualStreamToSingleStream)
        nn.Module.__init__(block)
        block.norm1 = other.norm1
        block.attn = TimmViT_CustomAttention_DualStreamToSingleStream.build_from_std_attn(other.attn)
        block.ls1 = other.ls1
        block.drop_path1 = other.drop_path1
        block.norm2 = other.norm2
        block.mlp = other.mlp
        block.ls2 = other.ls2
        block.drop_path2 = other.drop_path2
        return block


class TimmViT_CustomAttention_DualStreamToSingleStream(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, tokens_to_drop: int) -> torch.Tensor:
        B, N, C = x.shape

        x_s = torch.narrow(x, 1, tokens_to_drop, N - tokens_to_drop)
        q_s = self.q(x_s).reshape(B, N - tokens_to_drop, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q_s = self.q_norm(q_s)
        k = self.k_norm(k)

        if self.fused_attn:
            x_s = F.scaled_dot_product_attention(
                q_s, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q_s = q_s * self.scale
            attn = q_s @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_s = attn @ v

        x_s = x_s.transpose(1, 2).reshape(B, N - tokens_to_drop, C)
        x_s = self.proj(x_s)
        x_s = self.proj_drop(x_s)
        return x_s

    @classmethod
    def build_from_std_attn(cls, other: Attention):
        attn = cls.__new__(cls)
        nn.Module.__init__(attn)
        attn.attn_drop = other.attn_drop
        attn.proj_drop = other.proj_drop
        attn.scale = other.scale
        attn.num_heads = other.num_heads
        qkv_bias = other.qkv.bias is not None
        attn.q = nn.Linear(other.qkv.in_features, other.qkv.in_features, bias=qkv_bias)
        attn.kv = nn.Linear(other.qkv.in_features, other.qkv.in_features * 2, bias=qkv_bias)
        attn.q.weight.data.copy_(other.qkv.weight.data[:other.qkv.in_features])
        if qkv_bias:
            attn.q.bias.data.copy_(other.qkv.bias.data[:other.qkv.in_features])
        attn.kv.weight.data.copy_(other.qkv.weight.data[other.qkv.in_features:])
        if qkv_bias:
            attn.kv.bias.data.copy_(other.qkv.bias.data[other.qkv.in_features:])
        attn.proj = other.proj
        attn.fused_attn = other.fused_attn
        attn.q_norm = other.q_norm
        attn.k_norm = other.k_norm
        return attn


def drop_add_residual_stochastic_depth_dual_stream_to_single_stream(
    x: Tensor, drop_token_num: int,
    residual_func: Callable[[Tensor, int], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset, drop_token_num)

    x = torch.narrow(x, 1, drop_token_num, n - drop_token_num)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)
