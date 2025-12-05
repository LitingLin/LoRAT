from typing import Tuple, Sequence

import numpy as np
import torch

from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes

from .... import EvaluatorContext
from ....components.tensor_cache import CacheService, TensorCache


class TemplateFeatForegroundMaskGeneration:
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int], device: torch.device):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
        self.template_size = np.array(template_size, dtype=np.int64)
        self.device = device
        self.background_value = 0
        self.foreground_value = 1

    def start(self, context: EvaluatorContext):
        max_num_concurrent = context.max_batch_size * context.num_input_data_streams
        self.template_mask_cache = CacheService(TensorCache(max_num_concurrent,
                                                            (self.template_feat_size[1],
                                                             self.template_feat_size[0]),
                                                            self.device, torch.long))

    def stop(self):
        assert self.template_mask_cache.empty()
        del self.template_mask_cache

    def initialize(self, task_id: int, bbox: np.ndarray, cropping_parameter: np.ndarray):
        template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]),
                                   self.background_value, dtype=torch.long)
        template_cropped_bbox = get_foreground_bounding_box(bbox, cropping_parameter, self.template_size, self.stride)
        assert bbox_is_valid(template_cropped_bbox)
        template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
        template_mask[
            template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[
                2]] = self.foreground_value
        self.template_mask_cache.put(task_id, template_mask.to(self.device))

    def get_batch(self, task_ids: Sequence[int]):
        return self.template_mask_cache.get_batch(task_ids)

    def remove(self, task_id: int):
        self.template_mask_cache.delete(task_id)


def get_foreground_bounding_box(bbox: np.ndarray, siamfc_cropping_parameter: np.ndarray,
                                cropped_image_size: np.ndarray,
                                stride: Tuple[float, float]):
    siamfc_cropped_bbox = apply_siamfc_cropping_to_boxes(bbox, siamfc_cropping_parameter)
    bbox_clip_to_image_boundary_(siamfc_cropped_bbox, cropped_image_size)
    siamfc_cropped_bbox[0] /= stride[0]
    siamfc_cropped_bbox[1] /= stride[1]
    siamfc_cropped_bbox[2] /= stride[0]
    siamfc_cropped_bbox[3] /= stride[1]
    return bbox_rasterize(siamfc_cropped_bbox, dtype=np.int64)
