from trackit.miscellanies.printing import pretty_format
from trackit.miscellanies.torch.dtype import set_default_dtype
from trackit.models import ModelBuildContext, ModelImplementationSuggestions
from trackit.models.backbone.builder import build_backbone


def create_SPMTrack_build_context(config: dict):
    print('SPMTrack model config:\n' + pretty_format(config['model']))
    return ModelBuildContext(lambda impl_advice: build_SPMTrack_model(config, impl_advice),
                             lambda impl_advice: 'SPMTrack' + str(impl_advice.optimize_for_inference) + str(impl_advice.device) + str(impl_advice.dtype))


def build_SPMTrack_model(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'], model_impl_suggestions.load_pretrained,
                              device=model_impl_suggestions.device, dtype=model_impl_suggestions.dtype)
    model_type = model_config['type']
    with model_impl_suggestions.device, set_default_dtype(model_impl_suggestions.dtype):
        if model_type == 'dinov2':
            if not model_impl_suggestions.optimize_for_inference:
                from .SPMTrack import SPMTrack_DINOv2
                model = SPMTrack_DINOv2(backbone, common_config['template_feat_size'],
                                        common_config['search_region_feat_size'],
                                        model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                        model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                        model_config['tmoe']['expert_nums'], model_config['tmoe']['init_method'],
                                        shared_expert=model_config['tmoe']['shared_expert'], route_compression=model_config['tmoe']['route_compression'])
            else:
                from .SPMTrack_inference import SPMTrack_Inference_DINOv2
                model = SPMTrack_Inference_DINOv2(backbone, common_config['template_feat_size'],
                                                  common_config['search_region_feat_size'],
                                                  model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                                  model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                                  model_config['tmoe']['expert_nums'],
                                                  model_config['tmoe']['init_method'],
                                                  shared_expert=model_config['tmoe']['shared_expert'],
                                                  route_compression=model_config['tmoe']['route_compression'])
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model
