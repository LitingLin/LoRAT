from trackit.miscellanies.torch.dtype import set_default_dtype
from trackit.models import ModelBuildContext, ModelImplementationSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.printing import pretty_format


def get_LoRATv2_build_context(config: dict):
    print('LoRATv2 model config:\n' + pretty_format(config['model'], indent_level=1))
    return ModelBuildContext(lambda impl_advice: build_LoRATv2_model(config, impl_advice),
                             lambda impl_advice: 'LoRATv2' + str(impl_advice.device) + str(impl_advice.dtype) +
                                                 'inference' if impl_advice.optimize_for_inference else '' +
                                                 'flashattn' if config['model']['enable_flash_attn'] and not impl_advice.torch_jit_trace_compatible else '')

def build_LoRATv2_model(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'], model_impl_suggestions.load_pretrained,
                              device=model_impl_suggestions.device, dtype=model_impl_suggestions.dtype)

    template_size = common_config['templates'][0]['size']
    search_region_sizes = tuple(search_region['size'] for search_region in common_config['search_regions'])

    with model_impl_suggestions.device, set_default_dtype(model_impl_suggestions.dtype):
        from .model import LoRATv2
        trainable_bits = model_config['stream_specific_LoRA']['trainable_bits']
        if model_impl_suggestions.optimize_for_inference:
            trainable_bits = tuple(False for _ in trainable_bits)
        return LoRATv2(backbone, common_config['model_stride'],
                       template_size,
                       search_region_sizes,
                       model_config['with_cls_token'],
                       model_config['with_reg_token'],
                       trainable_bits,
                       model_config['stream_specific_LoRA']['lora']['r'],
                       model_config['stream_specific_LoRA']['lora']['alpha'],
                       model_config['stream_specific_LoRA']['lora']['dropout'],
                       model_config['enable_flash_attn'] and not model_impl_suggestions.torch_jit_trace_compatible)
