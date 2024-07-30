# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_bias = qkv_bias
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, relative_coords_table, relative_position_index, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.qkv_bias:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, relative_coords_table, relative_position_index, attn_mask, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        assert H % self.window_size == 0 and W % self.window_size == 0, "sizes of feature map must be multiples of window size"

        shift_size = self.shift_size
        if H == self.window_size and W == self.window_size:
            shift_size = 0
            assert attn_mask is None

        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, relative_coords_table, relative_position_index, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, H, W):
        flops = 0
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x, H // 2, W // 2

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def flops(self, H, W):
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        do_shift = True
        if min(input_resolution) <= window_size:
            window_size = min(input_resolution)
            do_shift = False
        self.window_size = window_size

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w],
            indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],
            indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, False)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) or not do_shift else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        attn_mask = None
        if H > self.window_size and H > self.window_size:
            attn_mask = self._get_attn_mask(H, W, x.device)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, self.relative_coords_table, self.relative_position_index, attn_mask, H, W)
            else:
                x = blk(x, self.relative_coords_table, self.relative_position_index, attn_mask, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def flops(self, H, W):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(H, W)
        if self.downsample is not None:
            flops += self.downsample.flops(H, W)
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def _get_attn_mask(self, H, W, device):
        attr_name = f'attn_mask_{H}_{W}'
        if hasattr(self, attr_name):
            return getattr(self, attr_name)

        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        shift_size = self.window_size // 2
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.register_buffer(attr_name, attn_mask.to(device), False)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, ape=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.ape = ape
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        Wh, Ww = (H // self.patch_size[0], W // self.patch_size[1])
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)

        if self.ape:
            absolute_pos_embed = self.absolute_pos_embed
            if Wh != self.num_patches[0] or Ww != self.num_patches[1]:
                absolute_pos_embed = absolute_pos_embed.transpose(1, 2).view(1, self.embed_dim, self.num_patches[0], self.num_patches[1])
                absolute_pos_embed = F.interpolate(absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
                absolute_pos_embed = absolute_pos_embed.transpose(1, 2).view(1, self.embed_dim, Wh * Ww)
            x = (x + absolute_pos_embed)

        return x, Wh, Ww

    def flops(self, H, W):
        Ho, Wo = (H // self.patch_size[0], W // self.patch_size[1])
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=256,
                 patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 out_norm=True):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, ape=ape)

        patches_resolution = self.patch_embed.patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        if out_norm:
            self.norm = norm_layer(self.num_features)
        self.out_norm = out_norm

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x):
        x, H, W = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        if self.out_norm:
            x = self.norm(x)  # B L C
        return x

    def flops(self, H, W):
        flops = 0
        flops += self.patch_embed.flops(H, W)
        patch_size = self.patch_embed.patch_size
        patches_resolution = (H // patch_size[0], W // patch_size[1])
        for i, layer in enumerate(self.layers):
            flops += layer.flops(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
        flops += self.num_features * patches_resolution[0] * patches_resolution[1] // (2 ** self.num_layers)
        return flops


_cfg = {
    'swinv2_base_patch4_window12_192_22k': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth',
        params=dict(img_size=192, patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32)), drop_path_rate=0.2
    ),
    'swinv2_large_patch4_window12_192_22k': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth',
        params=dict(img_size=192, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48)), drop_path_rate=0.2
    ),
    'swinv2_tiny_patch4_window8_256': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth',
        params=dict(img_size=256, patch_size=4, window_size=8, embed_dim=96, depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24)), drop_path_rate=0.2
    ),
    'swinv2_small_patch4_window8_256': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth',
        params=dict(img_size=256, patch_size=4, window_size=8, embed_dim=96, depths=(2, 2, 18, 2),
                    num_heads=(3, 6, 12, 24)), drop_path_rate=0.3
    ),
    'swinv2_base_patch4_window8_256': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth',
        params=dict(img_size=256, patch_size=4, window_size=8, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32)), drop_path_rate=0.5
    ),
    'swinv2_tiny_patch4_window16_256': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth',
        params=dict(img_size=256, patch_size=4, window_size=16, embed_dim=96, depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24)), drop_path_rate=0.2
    ),
    'swinv2_small_patch4_window16_256': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth',
        params=dict(img_size=256, patch_size=4, window_size=16, embed_dim=96, depths=(2, 2, 18, 2),
                    num_heads=(3, 6, 12, 24)), drop_path_rate=0.3
    ),
    'swinv2_base_patch4_window16_256': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth',
        params=dict(img_size=256, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32)), drop_path_rate=0.5
    ),
}


import copy


def _update_state_dict_(state_dict, num_layers, out_norm):
    for key in list(state_dict.keys()):
        if num_layers < 4:
            if key.startswith(f'layers.{num_layers - 1}.downsample'):
                state_dict.pop(key)
                continue
        if key.startswith('layers.'):
            layer_index = int(key[len('layers.'): len('layers.') + 1])
            if layer_index >= num_layers:
                state_dict.pop(key)
                continue
        if key.startswith('head'):
            state_dict.pop(key)
            continue
        if key.startswith('norm'):
            if num_layers != 4 or not out_norm:
                state_dict.pop(key)
                continue
        if 'relative_coords_table' in key:
            state_dict.pop(key)
            continue
        if 'relative_position_index' in key:
            state_dict.pop(key)
            continue
        if 'attn_mask' in key:
            state_dict.pop(key)
            continue


def build_swin_v2(name: str, load_pretrained=False, num_layers: int = 4, out_norm=False, **kwargs):
    cfg = copy.deepcopy(_cfg[name])

    cfg['params'].update(kwargs)
    cfg['params']['pretrained_window_sizes'] = (0, 0, 0, 0)
    assert 1 <= num_layers <= 4, 'num_layers must be between 1 and 4'

    if num_layers < 4:
        cfg['params']['depths'] = tuple(cfg['params']['depths'][i] for i in range(num_layers))
        cfg['params']['num_heads'] = tuple(cfg['params']['num_heads'][i] for i in range(num_layers))
        cfg['params']['pretrained_window_sizes'] = tuple(cfg['params']['pretrained_window_sizes'][i] for i in range(num_layers))

    cfg['params']['out_norm'] = out_norm

    model = SwinTransformerV2(**cfg['params'])
    if load_pretrained:
        state_dict = torch.hub.load_state_dict_from_url(cfg['url'], progress=True)
        state_dict = state_dict['model']
        _update_state_dict_(state_dict, num_layers, out_norm)
        model.load_state_dict(state_dict, strict=False)
    return model



if __name__ == '__main__':
    model = build_swin_v2('swinv2_base_patch4_window12_192_22k', load_pretrained=True, num_layers=3, img_size=384)

    print(model(torch.zeros((1, 3, 384, 384))).shape)
    print(model(torch.zeros((1, 3, 192, 192))).shape)
    # model(torch.zeros((1, 3, 128, 128)))
