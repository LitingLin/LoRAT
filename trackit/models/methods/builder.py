from torch import nn
from trackit.models import ModelBuildContext


def create_model_build_context(config: dict) -> ModelBuildContext:
    if config['type'] == 'LoRAT':
        from .LoRAT.builder import create_LoRAT_build_context
        build_context = create_LoRAT_build_context(config)
    elif config['type'] == 'LoRAT-timm':
        from .LoRAT.timm_vit.builder import create_LoRAT_build_context
        build_context = create_LoRAT_build_context(config)
    elif config['type'] == 'LoRAT-ablation':
        from .LoRAT.ablation.builder import create_LoRAT_build_context
        build_context = create_LoRAT_build_context(config)
    elif config['type'] == 'SPMTrack':
        from .SPMTrack.builder import create_SPMTrack_build_context
        build_context = create_SPMTrack_build_context(config)
    elif config['type'] == 'LoRATv2':
        from .LoRATv2.builder import get_LoRATv2_build_context
        build_context = get_LoRATv2_build_context(config)
    else:
        raise NotImplementedError(config['type'])
    if isinstance(build_context, nn.Module):
        model = build_context
        build_context = ModelBuildContext(lambda impl_suggestion: model.to(impl_suggestion.device, impl_suggestion.dtype),
                                          lambda impl_suggestion: model.__class__.__name__ + str(impl_suggestion.device) + str(impl_suggestion.dtype))
    return build_context
