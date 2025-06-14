from trackit.miscellanies.torch.dtype import set_default_dtype
from trackit.models import ModelBuildContext, ModelImplementationSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.printing import pretty_format


def create_LoRAT_build_context(config: dict):
    print('LoRAT model config:\n' + pretty_format(config['model'], indent_level=1))
    return ModelBuildContext(lambda impl_advice: build_LoRAT_model(config, impl_advice),
                             lambda impl_advice: get_LoRAT_build_string(config, impl_advice))


def build_LoRAT_model(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'], model_impl_suggestions.load_pretrained,
                              device=model_impl_suggestions.device, dtype=model_impl_suggestions.dtype)
    model_type = model_config['type']
    with model_impl_suggestions.device, set_default_dtype(model_impl_suggestions.dtype):
        if model_type == 'dinov2':
            from .lorat import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                 common_config['search_region_feat_size'])
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")

        if not _is_lora_enabled(config) or model_impl_suggestions.optimize_for_inference:
            from .funcs.vit_lora_utils import attach_lora_state_dict_hooks_
            attach_lora_state_dict_hooks_(model)
        else:
            from .funcs.vit_backbone_freeze import freeze_vit_backbone_
            from .funcs.vit_lora_utils import enable_lora_
            freeze_vit_backbone_(model)
            enable_lora_(model, model_config['lora']['r'], model_config['lora']['alpha'],
                         model_config['lora']['dropout'], model_config['lora']['use_rslora'])
    return model


def get_LoRAT_build_string(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_type = config['model']['type']
    build_string = 'LoRAT_' + model_type
    if _is_lora_enabled(config):
        build_string += '_lora'
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    build_string += '_' + str(model_impl_suggestions.dtype)
    build_string += '_' + str(model_impl_suggestions.device)
    if model_impl_suggestions.load_pretrained:
        build_string += '_pretrained'
    return build_string


def _is_lora_enabled(config: dict):
    lora_enable = 'lora' in config['model']
    if lora_enable:
        lora_enable = config['model']['lora'].get('enabled', True)
    return lora_enable

