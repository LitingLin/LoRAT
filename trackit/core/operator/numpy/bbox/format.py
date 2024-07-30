import numpy as np


def bbox_cxcywh_to_xyxy(bbox: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Converts bounding boxes from center (cx, cy) and size (w, h) format to (x_min, y_min, x_max, y_max) format.
    """
    xyxy_bbox = np.empty_like(bbox, dtype=dtype)
    w_2 = bbox[..., 2] / 2
    h_2 = bbox[..., 3] / 2
    xyxy_bbox[..., 0] = bbox[..., 0] - w_2
    xyxy_bbox[..., 1] = bbox[..., 1] - h_2
    xyxy_bbox[..., 2] = bbox[..., 0] + w_2
    xyxy_bbox[..., 3] = bbox[..., 1] + h_2
    return xyxy_bbox


def bbox_xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (x, y, width, height) format to (x_min, y_min, x_max, y_max) format.
    """
    xyxy_bbox = np.empty_like(bbox)
    xyxy_bbox[..., 0] = bbox[..., 0]
    xyxy_bbox[..., 1] = bbox[..., 1]
    xyxy_bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
    xyxy_bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
    return xyxy_bbox


def bbox_get_center_point(bbox: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Returns the center point (cx, cy) of the bounding box.
    """
    center = np.empty((*bbox.shape[:-1], 2), dtype=dtype)
    center[..., 0] = (bbox[..., 0] + bbox[..., 2]) * 0.5
    center[..., 1] = (bbox[..., 1] + bbox[..., 3]) * 0.5
    return center


def bbox_get_width_and_height(bbox: np.ndarray) -> np.ndarray:
    """
    Returns the width and height of the bounding box.
    """
    wh = np.empty((*bbox.shape[:-1], 2), dtype=bbox.dtype)
    wh[..., 0] = bbox[..., 2] - bbox[..., 0]
    wh[..., 1] = bbox[..., 3] - bbox[..., 1]
    return wh


def bbox_xyxy_to_cxcywh(bbox: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Converts bounding boxes from (x_min, y_min, x_max, y_max) format to center (cx, cy) and size (w, h) format.
    """
    cxcywh_bbox = np.empty_like(bbox, dtype=dtype)
    cxcywh_bbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2.
    cxcywh_bbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2.
    cxcywh_bbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    cxcywh_bbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return cxcywh_bbox


def bbox_xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (x_min, y_min, x_max, y_max) format to (x, y, width, height) format.
    """
    xywh_bbox = np.empty_like(bbox)
    xywh_bbox[..., 0] = bbox[..., 0]
    xywh_bbox[..., 1] = bbox[..., 1]
    xywh_bbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    xywh_bbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return xywh_bbox
