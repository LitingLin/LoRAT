import torch

from trackit.models.methods.builder import create_model_build_context
from trackit.models import ModelImplementationSuggestions
from trackit.miscellanies.torch.config_parsing.dtype import get_torch_dtype_with_machine_compatibility_checking
from trackit.miscellanies.torch.config_parsing.auto_mixed_precision import parse_auto_mixed_precision_config_with_machine_compatibility_checking


def build_vot_evaluation(config: dict, runtime_vars):
    from . import VOTEvaluationApplication
    device = torch.device(runtime_vars.device)
    dtype = get_torch_dtype_with_machine_compatibility_checking(
        config['run'].get('dtype', 'float32'), device.type)
    return VOTEvaluationApplication(create_model_build_context(config).create_fn(
        ModelImplementationSuggestions(device=device, dtype=dtype, optimize_for_inference=True)),
                                    config['common']['normalization'],
                                    device, dtype,
                                    parse_auto_mixed_precision_config_with_machine_compatibility_checking(
                                        config['run'].get('auto_mixed_precision', None), device.type),
                                    config['run'].get('visualize', False), runtime_vars.output_dir)
