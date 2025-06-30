import torch

from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.builder import \
    build_siamfc_search_region_cropping_parameter_provider_factory
from . import SPMTrack_EvaluationPipeline
from .plugin.builder import build_plugins
from ....components.post_process.builder import build_post_process
from ....components.segmentation.builder import build_segmentify_post_processor

def build_SPMTrack_pipeline(pipeline_config: dict, config: dict, device: torch.device):
    common_config = config['common']

    visualization = pipeline_config.get('visualization', False)
    print('pipeline: SPM tracker')
    print('visualization: ', visualization)

    plugins = build_plugins(pipeline_config['plugin'], config, device)

    return SPMTrack_EvaluationPipeline(
        device, common_config['template_size'], common_config['search_region_size'],
        build_siamfc_search_region_cropping_parameter_provider_factory(pipeline_config['search_region_cropping']),
        build_post_process(pipeline_config['post_process'], common_config, device),
        build_segmentify_post_processor(pipeline_config['segmentify'], common_config,
                                        device) if 'segmentify' in pipeline_config else None,
        common_config['interpolation_mode'], common_config['interpolation_align_corners'],
        common_config['normalization'],
        visualization, plugins)
