import torch

from trackit.models import ModelManager
from trackit.models.methods.builder import create_model_build_context
from trackit.miscellanies.torch.config_parsing.auto_mixed_precision import \
    parse_auto_mixed_precision_config_with_machine_compatibility_checking
from trackit.miscellanies.torch.config_parsing.dtype import get_torch_dtype_with_machine_compatibility_checking

from .model_profiling import ModelProfiler


def build_model_profiler(config: dict, runtime_vars, wandb_instance):
    model_profiler = None
    if config['run'].get('profiling', None) is not None and config['run']['profiling']['enabled']:
        device = torch.device(runtime_vars.device)
        profiling_config = config['run']['profiling']
        from trackit.models.utils.profiling import InferencePrecision
        def _parse_inference_precision_config(inference_precision_config: dict):
            dtype = get_torch_dtype_with_machine_compatibility_checking(inference_precision_config.get('dtype', torch.float32), device.type)
            amp_dtype = parse_auto_mixed_precision_config_with_machine_compatibility_checking(inference_precision_config.get('auto_mixed_precision'), device.type)
            return InferencePrecision(dtype, amp_dtype is not None, amp_dtype)
        inference_precision = _parse_inference_precision_config(profiling_config['eval'] if 'eval' in profiling_config else profiling_config)
        train_inference_precision = _parse_inference_precision_config(profiling_config['train'] if 'train' in profiling_config else profiling_config)
        model_profiler = ModelProfiler(torch.device(runtime_vars.device), inference_precision, train_inference_precision, wandb_instance)
    return model_profiler


def build_model_profiling_application(config: dict, runtime_vars,
                              wandb_instance):
    # build model factory
    model_manager = ModelManager(create_model_build_context(config))

    model_profiler = build_model_profiler(config, runtime_vars, wandb_instance)
    from .app import ModelProfilingApplication
    return ModelProfilingApplication(model_profiler, model_manager)
