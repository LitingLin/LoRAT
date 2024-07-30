import torch
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.builder import \
    build_siamfc_search_region_cropping_parameter_provider_factory
from ....components.post_process.builder import build_post_process
from ....components.segmentation.builder import build_segmentify_post_processor

from . import OneStreamTracker_Evaluation_MainPipeline


def build_one_stream_tracker_pipeline(pipeline_config: dict, config: dict, device: torch.device):
    common_config = config['common']

    visualization = pipeline_config.get('visualization', False)
    print('pipeline: one stream tracker')
    print('visualization: ', visualization)

    main_pipeline = OneStreamTracker_Evaluation_MainPipeline(
        device, common_config['template_size'],  common_config['search_region_size'],
        build_siamfc_search_region_cropping_parameter_provider_factory(pipeline_config['search_region_cropping']),
        build_post_process(pipeline_config['post_process'], common_config, device),
        build_segmentify_post_processor(pipeline_config['segmentify'], common_config,
                                        device) if 'segmentify' in pipeline_config else None,
        common_config['interpolation_mode'], common_config['interpolation_align_corners'],
        common_config['normalization'],
        visualization)

    pipelines = [main_pipeline]

    if 'plugin' in pipeline_config:
        pipelines.extend(_build_plugins(pipeline_config['plugin'], config, device))

    return pipelines


def _build_plugins(plugins_config, config, device):
    pipelines = []
    for plugin_config in plugins_config:
        if plugin_config['type'] == 'template_foreground_indicating_mask_generation':
            from .._common.template_foreground_indicating_mask_generation import TemplateFeatForegroundMaskGeneration
            pipelines.append(TemplateFeatForegroundMaskGeneration(config['common']['template_size'], config['common']['template_feat_size'], device))
        else:
            raise ValueError('Unknown plugin type: {}'.format(plugin_config['type']))
    return pipelines
