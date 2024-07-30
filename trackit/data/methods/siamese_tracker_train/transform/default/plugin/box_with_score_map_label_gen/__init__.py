import numpy as np
from typing import Iterable, Sequence, Mapping
import torch
from trackit.data.utils.collation_helper import collate_element_as_torch_tensor
from trackit.data.protocol.train_input import TrainData
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


def positive_sample_assignment(bbox: np.ndarray, response_map_size: np.ndarray, search_region_size: np.ndarray):
    '''

    :param bbox: (4,), in (xyxy) format
    :param response_map_size: (2,), response map size
    :param search_region_size: (2,), input search region size
    :return:
    '''
    scale = response_map_size / search_region_size
    indices = np.arange(0, response_map_size[0] * response_map_size[1], dtype=np.int64)
    indices = indices.reshape(response_map_size[1], response_map_size[0])
    scaled_bbox = bbox.copy()
    scaled_bbox[::2] = scaled_bbox[::2] * scale[0]
    scaled_bbox[1::2] = scaled_bbox[1::2] * scale[1]
    rasterized_scaled_bbox = bbox_rasterize(scaled_bbox, dtype=np.int64)
    positive_sample_indices = indices[rasterized_scaled_bbox[1]: rasterized_scaled_bbox[3],
                                      rasterized_scaled_bbox[0]: rasterized_scaled_bbox[2]].flatten()
    assert len(positive_sample_indices) > 0, (f'bbox is too small.\n'
                                              f'scale:\n{scale}\n'
                                              f'bbox:\n{bbox}\n'
                                              f'rasterized_scaled_bbox\n{rasterized_scaled_bbox}\n'
                                              f'scaled_bbox:\n{scaled_bbox}')
    return positive_sample_indices


class BoxWithScoreMapLabelGenerator:
    def __init__(self, response_map_size, search_region_size):
        self.response_map_size = np.array(response_map_size)
        self.search_region_size = np.array(search_region_size)

    def __call__(self, training_pair, context: dict, data: dict, _):
        if context['is_positive']:
            x_cropped_bbox = context['x_cropped_bbox']
            positive_sample_indices = positive_sample_assignment(x_cropped_bbox, self.response_map_size,
                                                                 self.search_region_size)
            normalized_bbox = x_cropped_bbox.copy()
            normalized_bbox[::2] = x_cropped_bbox[::2] / self.search_region_size[0]
            normalized_bbox[1::2] = x_cropped_bbox[1::2] / self.search_region_size[1]
            normalized_bbox = normalized_bbox.astype(np.float32)

            data['label'] = positive_sample_indices, normalized_bbox
        else:
            data['label'] = None, np.full((0, 4), np.nan, dtype=np.float32)


def _batch_collate_positive_sample_indices(positive_sample_indices_list: Iterable[np.ndarray]):
    collated_batch_ids = []
    collated_positive_sample_indices = []
    num_positive_samples = 0
    for index, positive_sample_indices in enumerate(positive_sample_indices_list):
        if positive_sample_indices is None:
            continue

        collated_batch_ids.append(torch.full((len(positive_sample_indices),), index, dtype=torch.long))
        collated_positive_sample_indices.append(torch.from_numpy(positive_sample_indices).to(torch.long))
        num_positive_samples += len(positive_sample_indices)

    num_positive_samples = torch.as_tensor((num_positive_samples,), dtype=torch.float)
    if num_positive_samples > 0:
        return torch.cat(collated_batch_ids), torch.cat(collated_positive_sample_indices), num_positive_samples
    else:
        return None, None, num_positive_samples


def box_with_score_map_label_collator(batch: Sequence[Mapping], collated: TrainData):
    label_list = tuple(data['label'] for data in batch)
    collated_batch_ids, collated_positive_sample_indices, num_positive_samples = \
        _batch_collate_positive_sample_indices(label[0] for label in label_list)
    collated_gt_bboxes = collate_element_as_torch_tensor(label_list, 1)

    collated.target.update({'num_positive_samples': num_positive_samples,
                            'positive_sample_batch_dim_indices': collated_batch_ids,
                            'positive_sample_map_dim_indices': collated_positive_sample_indices,
                            'boxes': collated_gt_bboxes})
