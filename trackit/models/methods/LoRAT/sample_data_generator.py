import torch
from typing import Tuple

from trackit.models import SampleInputDataGeneratorInterface


class LoRAT_DummyDataGenerator(SampleInputDataGeneratorInterface):
    def __init__(self, template_size: Tuple[int, int], search_region_size: Tuple[int, int], template_feat_map_size: Tuple[int, int]):
        self._template_size = template_size
        self._search_region_size = search_region_size
        self._template_feat_map_size = template_feat_map_size

    def get(self, batch_size: int, device: torch.device):
        return {'z': torch.full((batch_size, 3, self._template_size[1], self._template_size[0]), 0.5, device=device),
                'x': torch.full((batch_size, 3, self._search_region_size[1], self._search_region_size[0]), 0.5, device=device),
                'z_feat_mask': torch.full((batch_size, self._template_feat_map_size[1], self._template_feat_map_size[0]), 1, dtype=torch.long, device=device)}


def build_sample_input_data_generator(config: dict):
    common_config = config['common']
    return LoRAT_DummyDataGenerator(common_config['template_size'], common_config['search_region_size'], common_config['template_feat_size'])
