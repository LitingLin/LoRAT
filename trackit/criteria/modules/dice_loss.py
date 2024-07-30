# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from
#   https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
#   https://github.com/facebookresearch/detr/blob/master/models/segmentation.py
import torch


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss
