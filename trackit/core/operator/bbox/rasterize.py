import torch


def rasterize_bbox_torch_(bbox: torch.Tensor, eps: float = 1e-4):
    assert bbox.dtype in (torch.float32, torch.float64)
    bbox[..., 2] += (1 - eps)
    bbox[..., 3] += (1 - eps)
    torch.floor_(bbox)
    return bbox
