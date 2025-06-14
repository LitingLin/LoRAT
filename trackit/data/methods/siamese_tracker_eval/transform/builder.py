import torch

from trackit.miscellanies.printing import pretty_format


def build_data_transform(transform_config: dict, config: dict,
                         device: torch.device = torch.device('cpu'),
                         dtype: torch.dtype = torch.float32):
    common_config = config['common']
    print('transform config:\n' + pretty_format(transform_config, indent_level=1))
    if transform_config['type'] == 'default':
        from .default import SiameseTrackerEval_DefaultDataTransform
        return SiameseTrackerEval_DefaultDataTransform(common_config['template_size'],
                                                       transform_config['template_area_factor'],
                                                       transform_config.get('with_full_template_image', False),
                                                       common_config['interpolation_mode'],
                                                       common_config['interpolation_align_corners'],
                                                       common_config['normalization'],
                                                       device, dtype)
    elif transform_config['type'] == 'plain':
        from .plain import SiameseTrackerEval_PlainDataTransform
        return SiameseTrackerEval_PlainDataTransform(common_config['normalization'], device, dtype)
    else:
        raise ValueError(f"Unsupported transform type: {transform_config['type']}")
