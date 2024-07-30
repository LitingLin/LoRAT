# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn as nn
import math


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def iou_loss(pred: Tensor,
             target: Tensor,
             mode: str = 'log',
             eps: float = 1e-6) -> Tensor:
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)

    if fp16:
        ious = ious.to(torch.float16)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


def bounded_iou_loss(pred: Tensor,
                     target: Tensor,
                     beta: float = 0.2,
                     eps: float = 1e-3) -> Tensor:
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        beta (float, optional): Beta parameter in smoothl1.
        eps (float, optional): Epsilon to avoid NaN values.

    Return:
        Tensor: Loss tensor.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    # view(..., -1) does not work for empty tensor
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).flatten(1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)

    if fp16:
        gious = gious.to(torch.float16)

    loss = 1 - gious
    return loss


def diou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


def ciou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)

    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss


def eiou_loss(pred: Tensor,
              target: Tensor,
              smooth_point: float = 0.1,
              eps: float = 1e-7) -> Tensor:
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1.
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    # Intersection
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (
        ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (
            iy1 - ey1)
    # Union
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (
        ty2 - ty1) - intersection + eps
    # IoU
    ious = 1 - (intersection / union)

    # Smooth-EIoU
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point)
    return loss


def siou_loss(pred, target, eps=1e-7, neg_gamma=False):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    # modified clamp threshold zero to eps to avoid NaN
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # angle cost
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps

    sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    # distance cost
    rho_x = (s_cw / cw)**2
    rho_y = (s_ch / ch)**2

    # `neg_gamma=True` follows original implementation in paper
    # but setting `neg_gamma=False` makes training more stable.
    gamma = angle_cost - 2 if neg_gamma else 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # shape cost
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
        1 - torch.exp(-1 * omiga_h), 4)

    # SIoU
    sious = ious - 0.5 * (distance_cost + shape_cost)
    loss = 1 - sious.clamp(min=-1.0, max=1.0)
    return loss


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 eps: float = 1e-6,
                 mode: str = 'log') -> None:
        super().__init__()
        assert mode in ['linear', 'square', 'log']
        self.mode = mode
        self.eps = eps

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Return:
            Tensor: Loss tensor.
        """
        return iou_loss(
            pred,
            target,
            mode=self.mode,
            eps=self.eps)


class BoundedIoULoss(nn.Module):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        beta (float, optional): Beta parameter in smoothl1.
        eps (float, optional): Epsilon to avoid NaN values.
    """

    def __init__(self,
                 beta: float = 0.2,
                 eps: float = 1e-3) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Returns:
            Tensor: Loss tensor.
        """
        return bounded_iou_loss(
            pred,
            target,
            beta=self.beta,
            eps=self.eps)


class GIoULoss(nn.Module):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
    """

    def __init__(self,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Returns:
            Tensor: Loss tensor.
        """
        return giou_loss(
            pred,
            target,
            eps=self.eps)


class DIoULoss(nn.Module):
    r"""Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
    """

    def __init__(self,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Returns:
            Tensor: Loss tensor.
        """
        return diou_loss(
            pred,
            target,
            eps=self.eps)


class CIoULoss(nn.Module):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
    """

    def __init__(self,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Returns:
            Tensor: Loss tensor.
        """
        return ciou_loss(
            pred,
            target,
            eps=self.eps)


class EIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.smooth_point = smooth_point

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Returns:
            Tensor: Loss tensor.
        """
        return eiou_loss(
            pred,
            target,
            smooth_point=self.smooth_point,
            eps=self.eps)


class SIoULoss(nn.Module):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 neg_gamma: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.neg_gamma = neg_gamma

    def forward(self,
                pred: Tensor,
                target: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).

        Returns:
            Tensor: Loss tensor.
        """
        return siou_loss(
            pred,
            target,
            neg_gamma=self.neg_gamma)
