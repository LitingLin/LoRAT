import torch

from trackit.miscellanies.pretty_format import pretty_format
from .default import SiameseTrackerEval_DefaultDataTransform


def build_data_transform(transform_config: dict, config: dict, device: torch.device = torch.device('cpu')):
    common_config = config['common']
    print('transform config:\n' + pretty_format(transform_config))
    if transform_config['type'] == 'default':
        # currently only one type of transform is supported
        return SiameseTrackerEval_DefaultDataTransform(common_config['template_size'],
                                                       transform_config['template_area_factor'],
                                                       transform_config.get('with_full_template_image', False),
                                                       common_config['interpolation_mode'],
                                                       common_config['interpolation_align_corners'],
                                                       common_config['normalization'],
                                                       device)
    else:
        raise ValueError(f"Unsupported transform type: {transform_config['type']}")
