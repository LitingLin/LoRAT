from trackit.miscellanies.torch.dtype import set_default_dtype
from trackit.models import ModelBuildContext, ModelImplementationSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.printing import pretty_format
from ..funcs.vit_backbone_freeze import freeze_vit_backbone_
from ..funcs.vit_lora_utils import attach_lora_state_dict_hooks_, enable_lora_


def create_LoRAT_build_context(config: dict):
    print('LoRAT model config:\n' + pretty_format(config['model']))
    return ModelBuildContext(lambda impl_advice: build_LoRAT_model(config, impl_advice),
                             lambda impl_advice: get_LoRAT_build_string(config, impl_advice))


def build_LoRAT_model(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'], load_pretrained=model_impl_suggestions.load_pretrained,
                              device=model_impl_suggestions.device, dtype=model_impl_suggestions.dtype)
    model_type = model_config['type']
    with model_impl_suggestions.device, set_default_dtype(model_impl_suggestions.dtype):
        if model_type == 'dinov2_ia3':
            from .peft.ia3 import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                 common_config['search_region_feat_size'])
        elif model_type == 'dinov2_vpt_deep':
            from .peft.vpt_deep import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                 common_config['search_region_feat_size'],
                                 model_config['num_vpt_tokens'])
        elif model_type == 'dinov2_adapter':
            from .peft.adapter import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                 common_config['search_region_feat_size'],
                                 model_config['adapter_reduction_factor'])
        elif model_type == 'dinov2_lora_ablation':
            if not _is_lora_enabled(config) or model_impl_suggestions.optimize_for_inference:
                from ..lorat import LoRAT_DINOv2
                model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
                attach_lora_state_dict_hooks_(model)
            else:
                from .lora_ablation import LoRAT_DINOv2
                model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                     common_config['search_region_feat_size'],
                                     model_config['lora']['r'], model_config['lora']['alpha'],
                                     model_config['lora']['dropout'],
                                     model_config['lora']['use_rslora'], model_config['lora']['init_method'],
                                     model_config['lora']['target_modules']['q'],
                                     model_config['lora']['target_modules']['k'],
                                     model_config['lora']['target_modules']['v'],
                                     model_config['lora']['target_modules']['o'],
                                     model_config['lora']['target_modules']['mlp'])
        elif model_type == 'dinov2_input_embedding_ablation':
            from .input_emb.lorat_input_emb_ablation import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                 common_config['search_region_feat_size'],
                                 model_config['enable_token_type_embed'],
                                 model_config['enable_template_foreground_indicating_embed'])
            if not _is_lora_enabled(config) or model_impl_suggestions.optimize_for_inference:
                attach_lora_state_dict_hooks_(model)
            else:
                model.freeze_for_peft(model_config['pos_embed_trainable'])
                enable_lora_(model, model_config['lora']['r'], model_config['lora']['alpha'],
                             model_config['lora']['dropout'], model_config['lora']['use_rslora'])
        elif model_type == 'dinov2_sinusoidal':
            from .input_emb.lorat_pos_emb_sinusoidal import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
            if not _is_lora_enabled(config) or model_impl_suggestions.optimize_for_inference:
                attach_lora_state_dict_hooks_(model)
            else:
                model.freeze_for_peft()
                enable_lora_(model, model_config['lora']['r'], model_config['lora']['alpha'],
                             model_config['lora']['dropout'], model_config['lora']['use_rslora'])
        elif model_type == 'dinov2_sep_pos_emb':
            from .input_emb.lorat_sep_pos_emb import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
            if not _is_lora_enabled(config) or model_impl_suggestions.optimize_for_inference:
                attach_lora_state_dict_hooks_(model)
            else:
                model.freeze_for_peft(model_config['pos_embed_trainable'])
                enable_lora_(model, model_config['lora']['r'], model_config['lora']['alpha'],
                             model_config['lora']['dropout'], model_config['lora']['use_rslora'])
        else:
            if model_type == 'dinov2_mixattn':
                from .mixattn import LoRAT_DINOv2
                model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
            elif model_type == 'dinov2_convhead':
                from .head.lorat_conv_head import LoRAT_DINOv2
                model = LoRAT_DINOv2(backbone, model_config['head_channels'],
                                     common_config['template_feat_size'], common_config['search_region_feat_size'])
            elif model_type == 'dinov2_ostrackhead':
                from .head.lorat_ostrack_head import LoRAT_DINOv2
                model = LoRAT_DINOv2(backbone, model_config['head_channels'],
                                     common_config['template_feat_size'], common_config['search_region_feat_size'])
            else:
                raise NotImplementedError(f"Model type '{model_type}' is not supported.")

            if not _is_lora_enabled(config) or model_impl_suggestions.optimize_for_inference:
                attach_lora_state_dict_hooks_(model)
            else:
                freeze_vit_backbone_(model, freeze_out_norm=False)
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
