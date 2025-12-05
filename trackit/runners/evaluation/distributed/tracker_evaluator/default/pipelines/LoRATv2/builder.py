import torch

from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.builder import build_post_process
from .search_region_cropping import SiamFCCroppingParameterProvider

def build_LoRATv2_tracking_pipeline(pipeline_config: dict, config: dict, device: torch.device):
    common_config = config['common']
    template_size = common_config['templates'][0]['size']
    search_region_size = common_config['search_regions'][-1]['size']
    search_region_cropping_area_factor = common_config['search_regions'][-1]['area_factor']
    search_curation_parameter_provider_factory = lambda : SiamFCCroppingParameterProvider(search_region_cropping_area_factor,
                                                                                          pipeline_config['search_region_cropping'].get('min_object_size', 0))
    template_feature_map_size = template_size[0] * common_config['model_stride'], template_size[1] * common_config['model_stride']
    model_post_process = build_post_process(pipeline_config['post_process'], common_config, device)
    if pipeline_config['phase'] == 1:
        assert len(common_config['search_regions']) == 1
        from .phase_1 import LoRATv2_Phase_1_TrackingPipeline
        return LoRATv2_Phase_1_TrackingPipeline(device,
                                                template_size, search_region_size,
                                                search_curation_parameter_provider_factory,
                                                template_feature_map_size, model_post_process,
                                                common_config['interpolation_mode'],
                                                common_config['interpolation_align_corners'],
                                                common_config['normalization'])
    elif pipeline_config['phase'] == 2:
        assert len(common_config['search_regions']) == 2
        curated_dynamic_reference_frame_size = common_config['search_regions'][0]['size']
        dynamic_reference_frame_cropping_area_factor = common_config['search_regions'][0]['area_factor']
        dynamic_reference_frame_update_threshold = pipeline_config['dynamic_reference_frame_update_threshold']
        from .phase_2 import LoRATv2_Phase_2_TrackingPipeline
        return LoRATv2_Phase_2_TrackingPipeline(device,
                                                template_size, search_region_size,
                                                curated_dynamic_reference_frame_size,
                                                search_curation_parameter_provider_factory,
                                                dynamic_reference_frame_cropping_area_factor,
                                                dynamic_reference_frame_update_threshold,
                                                template_feature_map_size, model_post_process,
                                                common_config['interpolation_mode'],
                                                common_config['interpolation_align_corners'],
                                                common_config['normalization'])
    else:
        raise ValueError(f'Unknown phase: {pipeline_config["phase"]}')
