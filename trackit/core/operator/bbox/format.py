import torch


def bbox_xyxy_to_cxcywh_torch(x: torch.Tensor):
    x0, y0, x1, y1 = x.unbind(-1)
    out = torch.empty_like(x)
    out[..., 0] = (x0 + x1) / 2
    out[..., 1] = (y0 + y1) / 2
    out[..., 2] = (x1 - x0)
    out[..., 3] = (y1 - y0)
    return out


def bbox_cxcywh_to_xyxy_torch(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    out = torch.empty_like(x)
    half_w = 0.5 * w
    half_h = 0.5 * h
    out[..., 0] = (x_c - half_w)
    out[..., 1] = (y_c - half_h)
    out[..., 2] = (x_c + half_w)
    out[..., 3] = (y_c + half_h)
    return out


def bbox_xyxy_to_xywh_torch(box: torch.Tensor):
    x0, y0, x1, y1 = box.unbind(-1)
    out = torch.empty_like(box)
    out[..., 0] = x0
    out[..., 1] = y0
    out[..., 2] = x1 - x0
    out[..., 3] = y1 - y0
    return out
