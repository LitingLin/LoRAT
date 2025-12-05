from typing import Tuple, Sequence, Mapping
import numpy as np

from trackit.data.protocol.train_input import TrainData
from trackit.data.utils.collation_helper import collate_element_as_torch_tensor
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize

from .. import ExtraTransform


class TemplateFeatMaskGenerator(ExtraTransform):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int]):
        self.template_size = template_size
        self.template_feat_size = template_feat_size

    def __call__(self, training_pair, context: dict, data: dict, _):
        data['z_cropped_bbox_feat_map_mask'] = _generate_feat_mask(context[f'z_0_cropped_bbox'],
                                                                   self.template_size,
                                                                   self.template_feat_size)

    @staticmethod
    def collate(batch: Sequence[Mapping], collated: TrainData):
        collated.input['z_feat_mask'] = collate_element_as_torch_tensor(batch,
                                                                        'z_cropped_bbox_feat_map_mask')


def _generate_feat_mask(bbox: np.ndarray, input_size: Tuple[int, int], feat_size: Tuple[int, int]):
    feat_bbox = bbox.copy()
    stride = (input_size[0] / feat_size[0], input_size[1] / feat_size[1])
    feat_bbox[0] /= stride[0]
    feat_bbox[1] /= stride[1]
    feat_bbox[2] /= stride[0]
    feat_bbox[3] /= stride[1]
    feat_bbox = bbox_rasterize(feat_bbox, dtype=np.int64)

    feat_mask = np.full((feat_size[1], feat_size[0]), 0, dtype=np.int64)
    feat_mask[feat_bbox[1]: feat_bbox[3], feat_bbox[0]: feat_bbox[2]] = True
    return feat_mask
