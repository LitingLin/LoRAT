from typing import Tuple, Sequence, Mapping
import numpy as np
import torch

from trackit.data.protocol.train_input import TrainData
from trackit.data.utils.collation_helper import collate_element_as_torch_tensor
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


class TemplateFeatMaskGenerator:
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int]):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = (self.template_size[0] / self.template_feat_size[0], self.template_size[1] / self.template_feat_size[1])
        self.background_value = 0
        self.foreground_value = 1

    def __call__(self, training_pair, context: dict, data: dict, _, length_z: int, length_x: int):
        for i in range(length_z):
            z_cropped_bbox = context[f'z_{i}_cropped_bbox']
            mask = np.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value, dtype=np.int64)
            z_cropped_bbox = z_cropped_bbox.copy()
            z_cropped_bbox[0] /= self.stride[0]
            z_cropped_bbox[1] /= self.stride[1]
            z_cropped_bbox[2] /= self.stride[0]
            z_cropped_bbox[3] /= self.stride[1]
            z_cropped_bbox = bbox_rasterize(z_cropped_bbox, dtype=np.int64)
            mask[z_cropped_bbox[1]:z_cropped_bbox[3], z_cropped_bbox[0]:z_cropped_bbox[2]] = self.foreground_value
            data[f'z_{i}_cropped_bbox_feat_map_mask'] = mask


def template_feat_mask_data_collator(batch: Sequence[Mapping], collated: TrainData, max_z: int, max_x: int):
    for i in range(max_z + 1):
        collated.input[f'z_{i}_feat_mask'] = collate_element_as_torch_tensor(batch, f'z_{i}_cropped_bbox_feat_map_mask')
