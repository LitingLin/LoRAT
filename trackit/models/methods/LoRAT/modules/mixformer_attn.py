from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from trackit.models.backbone.dinov2.layers.attention import Attention
from trackit.models.backbone.dinov2.layers.block import Block
from trackit.models.backbone.dinov2.layers import Mlp
from trackit.models.backbone.dinov2.layers.drop_path import DropPath
from trackit.models.backbone.dinov2.layers.layer_scale import LayerScale


class MixformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = False

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)

        # asymmetric mixed attention
        if self.fused_attn:
            x_mt = F.scaled_dot_product_attention(
                q_mt, k_mt, v_mt,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q_mt * self.scale) @ k_mt.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_mt = attn @ v_mt
        x_mt = x_mt.transpose(1, 2).reshape(B, t_h * t_w, C)

        if self.fused_attn:
            x_s = F.scaled_dot_product_attention(
                q_s, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q_s * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_s = attn @ v
        x_s = x_s.transpose(1, 2).reshape(B, s_h * s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @classmethod
    def build_from_std_attn(cls, other: Attention):
        attn = cls.__new__(cls)
        nn.Module.__init__(attn)
        attn.attn_drop = other.attn_drop
        attn.proj_drop = other.proj_drop
        attn.scale = other.scale
        attn.num_heads = other.num_heads
        attn.qkv = other.qkv
        attn.proj = other.proj
        attn.fused_attn = other.fused_attn
        return attn


class MixFormerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = Attention,
            ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, *args) -> Tensor:
        def attn_residual_func(x: Tensor, *args) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), *args))

        def ffn_residual_func(x: Tensor, *args) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, args,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x, args,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, *args))
            x = x + self.drop_path2(ffn_residual_func(x, *args))
        else:
            x = x + attn_residual_func(x, *args)
            x = x + ffn_residual_func(x, *args)
        return x

    @classmethod
    def build_from_std_block(cls, other: Block):
        block = cls.__new__(cls)
        nn.Module.__init__(block)
        block.norm1 = other.norm1
        block.attn = MixformerAttention.build_from_std_attn(other.attn)
        block.ls1 = other.ls1
        block.drop_path1 = other.drop_path1
        block.norm2 = other.norm2
        block.mlp = other.mlp
        block.ls2 = other.ls2
        block.drop_path2 = other.drop_path2
        block.sample_drop_ratio = other.sample_drop_ratio
        return block


def drop_add_residual_stochastic_depth(
        x: Tensor, o,
        residual_func: Callable[[Tensor], Tensor],
        sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset, *o)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)
