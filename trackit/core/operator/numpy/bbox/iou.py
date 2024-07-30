import numpy as np
from .intersection import bbox_compute_intersection_area, bbox_compute_area


def bbox_compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray):
    intersection_area = bbox_compute_intersection_area(bbox_a, bbox_b)
    union_area = bbox_compute_area(bbox_a) + bbox_compute_area(bbox_b) - intersection_area
    return intersection_area / union_area
