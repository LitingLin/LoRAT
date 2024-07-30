from trackit.miscellanies.pretty_format import pretty_format
from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from ..sample_data_generator import build_sample_input_data_generator


def get_LoRAT_build_context(config: dict):
    print('LoRAT model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_LoRAT_model(config, impl_advice),
                                lambda impl_advice: get_LoRAT_build_string(config['model']['type'], impl_advice),
                                build_sample_input_data_generator(config))


def build_LoRAT_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    if model_type == 'dinov2_mixattn':
        if model_impl_suggestions.optimize_for_inference:
            from .mixattn.baseline_mixattn import LoRATBaseline_DINOv2
            model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
        else:
            from .mixattn.lorat_mixattn import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['lora']['r'], model_config['lora']['alpha'], model_config['lora']['dropout'],
                                 model_config['lora']['use_rslora'])
    elif model_type == 'dinov2_lora_ablation':
        if model_impl_suggestions.optimize_for_inference:
            from ..lorat_full_finetune import LoRATBaseline_DINOv2
            model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
        else:
            from .lora_conf.lorat import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['lora']['r'], model_config['lora']['alpha'], model_config['lora']['dropout'],
                                 model_config['lora']['use_rslora'], model_config['lora']['init_method'],
                                 model_config['lora']['target_modules']['q'],
                                 model_config['lora']['target_modules']['k'],
                                 model_config['lora']['target_modules']['v'],
                                 model_config['lora']['target_modules']['o'],
                                 model_config['lora']['target_modules']['mlp'])
    elif model_type == 'dinov2_full_finetune_mixattn':
        from .mixattn.baseline_mixattn import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_full_finetune_convhead':
        from .head.lorat_full_finetune_conv_head import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, model_config['head_channels'],
                                     common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_convhead':
        if model_impl_suggestions.optimize_for_inference:
            from .head.lorat_full_finetune_conv_head import LoRATBaseline_DINOv2
            model = LoRATBaseline_DINOv2(backbone, model_config['head_channels'],
                                         common_config['template_feat_size'], common_config['search_region_feat_size'])
        else:
            from .head.lorat_conv_head import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, model_config['head_channels'],
                                 common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_full_finetune_ostrackhead':
        from .head.lorat_full_finetune_ostrack_head import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, model_config['head_channels'],
                                     common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_ostrackhead':
        if model_impl_suggestions.optimize_for_inference:
            from .head.lorat_full_finetune_ostrack_head import LoRATBaseline_DINOv2
            model = LoRATBaseline_DINOv2(backbone, model_config['head_channels'],
                                         common_config['template_feat_size'], common_config['search_region_feat_size'])
        else:
            from .head.lorat_ostrack_head import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, model_config['head_channels'],
                                 common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_full_finetune_sinusoidal':
        from .input_emb.lorat_full_finetune_sinusoidal import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_full_finetune_sep_pos_emb':
        from .input_emb.lorat_full_finetune_sep_pos_emb import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_full_finetune_input_embedding_ablation':
        from .input_emb.lorat_full_finetune_input_emb_ablation import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'],
                                     common_config['search_region_feat_size'],
                                     model_config['enable_token_type_embed'],
                                     model_config['enable_template_foreground_indicating_embed'])
    elif model_type == 'dinov2_input_embedding_ablation':
        if model_impl_suggestions.optimize_for_inference:
            from .input_emb.lorat_full_finetune_input_emb_ablation import LoRATBaseline_DINOv2
            model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'],
                                         common_config['search_region_feat_size'],
                                         model_config['enable_token_type_embed'],
                                         model_config['enable_template_foreground_indicating_embed'])
        else:
            from .input_emb.lorat_input_emb_ablation import LoRAT_DINOv2
            model = LoRAT_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['pos_embed_trainable'],
                                 model_config['enable_token_type_embed'],
                                 model_config['enable_template_foreground_indicating_embed'],
                                 model_config['lora']['r'], model_config['lora']['alpha'],
                                 model_config['lora']['dropout'], model_config['lora']['use_rslora'])
    elif model_type == 'dinov2_ia3':
        from .peft.ia3 import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    elif model_type == 'dinov2_vpt_deep':
        from .peft.vpt_deep import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                     model_config['num_vpt_tokens'])
    elif model_type == 'dinov2_adapter':
        from .peft.adapter import LoRATBaseline_DINOv2
        model = LoRATBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                     model_config['adapter_reduction_factor'])
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model


def get_LoRAT_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'LoRAT'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string
