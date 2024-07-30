import torch

from trackit.miscellanies.pretty_format import pretty_format
from . import MaskGenerator
from .wrapper import Segmentify_PostProcessor
from trackit.miscellanies.torch.data_parallel import should_use_data_parallel


def _build_mask_generator(mask_generator_config: dict, common_config: dict, device: torch.device) -> MaskGenerator:
    mask_generator_type = mask_generator_config['type']
    if mask_generator_type == 'sam':
        from .sam import SAM_BoxToMaskPostProcess
        return SAM_BoxToMaskPostProcess(mask_generator_config['model_type'], mask_generator_config['model_name'],
                                        device,
                                        common_config['interpolation_mode'],
                                        common_config['interpolation_align_corners'],
                                        mask_generator_config['mask_threshold'],
                                        mask_generator_config.get('enable_data_parallel',
                                                                  False) and should_use_data_parallel(device))
    elif mask_generator_type == 'sam_hq':
        from .sam_hq import SAMHQ_BoxToMaskPostProcess
        return SAMHQ_BoxToMaskPostProcess(mask_generator_config['model_name'], device,
                                          common_config['interpolation_mode'],
                                          common_config['interpolation_align_corners'],
                                          mask_generator_config['mask_threshold'])
    else:
        raise NotImplementedError(f"{mask_generator_config['type']} Not implemented")


def build_segmentify_post_processor(segmentify_post_processor_config: dict, common_config: dict, device: torch.device
                                    ) -> Segmentify_PostProcessor:
    print('Segmentify post processor: \n' + pretty_format(segmentify_post_processor_config))
    mask_generator = _build_mask_generator(segmentify_post_processor_config['mask_generator'], common_config, device)
    mask_generator = Segmentify_PostProcessor(segmentify_post_processor_config['search_region_size'],
                                              segmentify_post_processor_config['area_factor'],
                                              device, mask_generator,
                                              common_config['interpolation_mode'],
                                              common_config['interpolation_align_corners'])
    return mask_generator
