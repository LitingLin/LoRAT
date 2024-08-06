from torch import nn
from trackit.models import ModelBuildingContext


def get_model_build_context(config: dict) -> ModelBuildingContext:
    if config['type'] == 'LoRAT':
        from .LoRAT.builder import get_LoRAT_build_context
        build_context = get_LoRAT_build_context(config)
    elif config['type'] == 'LoRAT-ablation':
        from .LoRAT.ablation.builder import get_LoRAT_build_context
        build_context = get_LoRAT_build_context(config)
    else:
        raise NotImplementedError()
    if isinstance(build_context, nn.Module):
        model = build_context
        build_context = ModelBuildingContext(lambda _: model, lambda _: model.__class__.__name__, None)
    return build_context
