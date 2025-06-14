import numpy as np
from typing import Union, Optional, Tuple

from timm.layers import to_2tuple

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary, is_bbox_intersecting_image
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params
from . import CroppingParameterProvider


def _adjust_bbox_size(bounding_box: np.ndarray, min_wh: np.ndarray):
    bounding_box = bbox_xyxy_to_cxcywh(bounding_box)
    bounding_box[2:4] = np.maximum(bounding_box[2:4], min_wh)
    return bbox_cxcywh_to_xyxy(bounding_box)


class SiamFCCroppingParameterSimpleProvider(CroppingParameterProvider):
    def __init__(self, area_factor: float, min_object_size: Optional[Union[Tuple[float, float], float]] = None):
        self.area_factor = area_factor
        self.cached_bbox = np.empty((4,), dtype=float)
        if min_object_size is not None:
            min_object_size = to_2tuple(min_object_size)
            min_object_size = np.array(min_object_size, dtype=float)
        self.min_object_size = min_object_size

    def initialize(self, bbox: np.ndarray):
        assert bbox_is_valid(bbox)
        self.cached_bbox[:] = bbox

    def get(self, output_image_size: np.ndarray, area_factor: float | None = None):
        bbox = self.cached_bbox
        if self.min_object_size is not None:
            bbox = _adjust_bbox_size(bbox, self.min_object_size)
        area_factor = area_factor if area_factor is not None else self.area_factor
        cropping_params = get_siamfc_cropping_params(bbox, area_factor, output_image_size)
        return cropping_params

    def update(self, predicted_confidence: Optional[float], predicted_bbox: np.ndarray, image_size: np.ndarray):
        assert image_size[0] > 0 and image_size[1] > 0

        predicted_bbox = bbox_clip_to_image_boundary(predicted_bbox, image_size)
        if not bbox_is_valid(predicted_bbox):
            return
        if not is_bbox_intersecting_image(predicted_bbox, image_size):
            return
        self.cached_bbox[:] = predicted_bbox
