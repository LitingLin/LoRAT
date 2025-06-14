from typing import Tuple
import torch


def generate_LoRAT_sample_data(z_feat_size: Tuple[int, int], x_feat_size: Tuple[int, int], patch_size: Tuple[int, int],
                               batch_size: int, device: torch.device, dtype: torch.dtype):
    z_feat_W, z_feat_H = z_feat_size
    x_feat_W, x_feat_H = x_feat_size
    patch_H, patch_W = patch_size

    z = torch.full((batch_size, 3, z_feat_H * patch_H, z_feat_W * patch_W), 0.5, dtype=dtype, device=device)
    x = torch.full((batch_size, 3, x_feat_H * patch_H, x_feat_W * patch_W), 0.5, dtype=dtype, device=device)
    z_feat_mask = torch.full((batch_size, z_feat_H, z_feat_W), 1, dtype=torch.long, device=device)

    return z, x, z_feat_mask
