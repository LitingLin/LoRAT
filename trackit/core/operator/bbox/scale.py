import torch


def bbox_scale(bbox: torch.Tensor, scale: torch.Tensor):
    """
    Args:
        bbox (torch.Tensor): (n, 4)
        scale (torch.Tensor): (n, 2)
    Returns:
        torch.Tensor: scaled torch tensor, (n, 4)
    """
    out_bbox = torch.empty_like(bbox)
    out_bbox[..., ::2] = bbox[..., ::2] * scale[..., (0, )]
    out_bbox[..., 1::2] = bbox[..., 1::2] * scale[..., (1, )]
    return out_bbox
