from typing import Tuple
import numpy as np
from timm.layers import to_2tuple

from trackit.core.runtime.build_context import BuildContext
from trackit.miscellanies.pretty_format import pretty_format
from .plugin.builder import build_plugins
from .augmentation.builder import build_augmentation_pipeline
from .processor import SiamTrackerTrainingPairProcessor, SiamTrackerTrainingPairProcessorBatchCollator, \
    SiamTrackerTrainingPairProcessorHostLoggingHook, SiamFCCroppingParameter


def _build_siamfc_cropping_parameter(siamfc_cropping_config: dict, output_size: Tuple[int, int],
                                     interpolation_mode: str,
                                     interpolation_align_corners: bool) -> SiamFCCroppingParameter:
    area_factor = siamfc_cropping_config['area_factor']
    scale_jitter_factor = siamfc_cropping_config.get('scale_jitter', 0.)
    translation_jitter_factor = siamfc_cropping_config.get('translation_jitter', 0.)
    output_min_object_size_in_pixel = np.array(to_2tuple(siamfc_cropping_config.get('min_object_size', (0., 0.))))
    output_max_object_size_in_pixel = np.array(to_2tuple(siamfc_cropping_config.get('max_object_size', (float("inf"), float("inf")))))
    output_min_object_size_in_ratio = siamfc_cropping_config.get('min_object_ratio', 0.)
    output_max_object_size_in_ratio = siamfc_cropping_config.get('max_object_ratio', 1.)
    return SiamFCCroppingParameter(np.array(output_size), area_factor, scale_jitter_factor, translation_jitter_factor,
                                   output_min_object_size_in_pixel, output_min_object_size_in_ratio,
                                   output_max_object_size_in_pixel, output_max_object_size_in_ratio,
                                   interpolation_mode, interpolation_align_corners)


def build_siamese_tracker_training_data_processing_components(transform_config: dict, config: dict, build_context: BuildContext):
    additional_processors, additional_data_collators, additional_host_data_pipelines = (
        build_plugins(transform_config, config, build_context))

    common_config = config['common']
    interpolation_mode = common_config['interpolation_mode']
    interpolation_align_corners = common_config['interpolation_align_corners']

    template_siamfc_cropping_parameter = \
        _build_siamfc_cropping_parameter(transform_config['SiamFC_cropping']['template'],
                                         common_config['template_size'],
                                         interpolation_mode, interpolation_align_corners)

    search_region_siamfc_cropping_parameter = \
        _build_siamfc_cropping_parameter(transform_config['SiamFC_cropping']['search_region'],
                                         common_config['search_region_size'],
                                         interpolation_mode, interpolation_align_corners)

    processor = SiamTrackerTrainingPairProcessor(
        template_siamfc_cropping_parameter,
        search_region_siamfc_cropping_parameter,
        build_augmentation_pipeline(transform_config['augmentation']),
        common_config['normalization'],
        additional_processors,
        transform_config.get('visualize', False))

    batch_collator = SiamTrackerTrainingPairProcessorBatchCollator(additional_data_collators)
    print('transform config:\n', pretty_format(transform_config))
    return processor, batch_collator, (
        SiamTrackerTrainingPairProcessorHostLoggingHook(), *additional_host_data_pipelines)
