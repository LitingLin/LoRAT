import numpy as np
from .area import bbox_compute_area


def bbox_get_intersection_region(bbox_a: np.ndarray, bbox_b: np.ndarray):
    intersection = np.concatenate((np.maximum(bbox_a[..., :2], bbox_b[..., :2]), np.minimum(bbox_a[..., 2:], bbox_b[..., 2:])), axis=-1)
    intersection[(intersection[..., 2] - intersection[..., 0]) <= 0, ...] = 0
    intersection[(intersection[..., 3] - intersection[..., 1]) <= 0, ...] = 0
    return intersection


def bbox_compute_intersection_area(bbox_a: np.ndarray, bbox_b: np.ndarray):
    intersection = bbox_get_intersection_region(bbox_a, bbox_b)
    intersection_area = bbox_compute_area(intersection)
    return intersection_area


def bbox_has_intersection(bbox_a: np.ndarray, bbox_b: np.ndarray):
    return bbox_compute_intersection_area(bbox_a, bbox_b) > 0.
