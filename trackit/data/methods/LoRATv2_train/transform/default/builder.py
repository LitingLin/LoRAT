import torch
from trackit.core.runtime.build_context import BuildContext
from trackit.miscellanies.printing import pretty_format
from .plugin.builder import build_plugins
from .augmentation.builder import build_augmentation_pipeline
from .processor import (DefaultTemporalTrackerProcessor, TemporalTrackerDefaultProcessorBatchCollator,
                        SiamFCCroppingParameter)



def _build_siamfc_cropping_parameter(cropping_config: dict,
                                     jittering_config: dict | None,
                                     interpolation_mode: str,
                                     interpolation_align_corners: bool):
    output_size = cropping_config['size']
    area_factor = cropping_config['area_factor']
    if jittering_config is None:
        jittering_config = {}
    return SiamFCCroppingParameter(output_size, area_factor, **jittering_config,
                                   interpolation_mode=interpolation_mode,
                                   interpolation_align_corners= interpolation_align_corners)


def _prase_siamfc_cropping_config(siamfc_cropping_config: list, siamfc_cropping_jittering_config: dict | list | None,
                                     interpolation_mode: str,
                                     interpolation_align_corners: bool):
    for index, sub_input_stream_config in enumerate(siamfc_cropping_config):
        if isinstance(siamfc_cropping_jittering_config, dict):
            yield _build_siamfc_cropping_parameter(sub_input_stream_config, siamfc_cropping_jittering_config,
                                                   interpolation_mode, interpolation_align_corners)
        elif isinstance(siamfc_cropping_jittering_config, list):
            yield _build_siamfc_cropping_parameter(sub_input_stream_config, siamfc_cropping_jittering_config[index],
                                                   interpolation_mode, interpolation_align_corners)
        elif siamfc_cropping_jittering_config is None:
            yield _build_siamfc_cropping_parameter(sub_input_stream_config, None,
                                                   interpolation_mode, interpolation_align_corners)
        else:
            raise ValueError('siamfc_cropping_jittering_config must be a dict or a list of dicts')



def build_siamese_tracker_training_data_processing_components(transform_config: dict, config: dict,
                                                              build_context: BuildContext, dtype: torch.dtype):
    additional_processors, additional_data_collators, additional_data_pipelines_on_main_process = (
        build_plugins(transform_config, config, build_context, dtype))

    common_config = config['common']
    interpolation_mode = common_config['interpolation_mode']
    interpolation_align_corners = common_config['interpolation_align_corners']

    template_siamfc_cropping_parameter = \
        _prase_siamfc_cropping_config(common_config['templates'],
                                         transform_config.get('SiamFC_cropping', {}).get('templates'),
                                         interpolation_mode, interpolation_align_corners)

    search_region_siamfc_cropping_parameter = \
        _prase_siamfc_cropping_config(common_config['search_regions'],
                                         transform_config.get('SiamFC_cropping', {}).get('search_regions'),
                                         interpolation_mode, interpolation_align_corners)

    augmentation_pipeline = build_augmentation_pipeline(transform_config['augmentation'])
    if 'static_image_augmentation' in transform_config:
        static_image_augmentation_pipeline = build_augmentation_pipeline(transform_config['static_image_augmentation'])
    else:
        static_image_augmentation_pipeline = augmentation_pipeline
    processor = DefaultTemporalTrackerProcessor(
        template_siamfc_cropping_parameter,
        search_region_siamfc_cropping_parameter,
        augmentation_pipeline, static_image_augmentation_pipeline,
        common_config['normalization'],
        additional_processors,
        transform_config.get('visualize', False))

    batch_collator = TemporalTrackerDefaultProcessorBatchCollator(dtype, additional_data_collators)
    print('transform config:\n' + pretty_format(transform_config, indent_level=1))
    return processor, batch_collator, additional_data_pipelines_on_main_process
