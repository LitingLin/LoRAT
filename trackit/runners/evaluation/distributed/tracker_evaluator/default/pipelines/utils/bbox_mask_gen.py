import numpy as np
from typing import Tuple
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


def get_foreground_bounding_box(bbox: np.ndarray, siamfc_cropping_parameter: np.ndarray, stride: Tuple[float, float]):
    siamfc_cropped_bbox = apply_siamfc_cropping_to_boxes(bbox, siamfc_cropping_parameter)
    siamfc_cropped_bbox[0] /= stride[0]
    siamfc_cropped_bbox[1] /= stride[1]
    siamfc_cropped_bbox[2] /= stride[0]
    siamfc_cropped_bbox[3] /= stride[1]
    return bbox_rasterize(siamfc_cropped_bbox, dtype=np.int64)
