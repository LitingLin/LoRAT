import numpy as np


def bbox_horizontal_flip(bbox: np.ndarray, width: int):
    bbox = bbox.copy()
    bbox[..., 0] = width - bbox[..., 0]
    bbox[..., 2] = width - bbox[..., 2]
    bbox = bbox[..., [2, 1, 0, 3]]
    return bbox


def bbox_vertical_flip(bbox: np.ndarray, height: int):
    bbox = bbox.copy()
    bbox[..., 1] = height - bbox[..., 1]
    bbox[..., 3] = height - bbox[..., 3]
    bbox = bbox[..., [0, 3, 2, 1]]
    return bbox


def bbox_diagonal_flip(bbox: np.ndarray, width: int, height: int):
    bbox = bbox.copy()
    bbox[..., 0] = width - bbox[..., 0]
    bbox[..., 2] = width - bbox[..., 2]
    bbox[..., 1] = height - bbox[..., 1]
    bbox[..., 3] = height - bbox[..., 3]
    bbox = bbox[..., [2, 3, 0, 1]]
    return bbox


def bbox_flip(bbox: np.ndarray, width: int, height: int, flip_horizontal: bool, flip_vertical: bool):
    if flip_horizontal:
        bbox = bbox_horizontal_flip(bbox, width)
    if flip_vertical:
        bbox = bbox_vertical_flip(bbox, height)
    return bbox
