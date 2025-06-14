from .optimizer.builder import build_optimizer
from .lr_scheduler.builder import build_lr_scheduler
from .wd_scheduler.builder import build_weight_decay_scheduler
from .parameter_updater.builder import build_parameter_updater
from typing import Optional
import torch
from functools import partial
from contextlib import nullcontext
from . import OptimizationModulesAndOptions
from ..torch_compile import TorchCompileOptions


def build_default_optimization_modules(model: torch.nn.Module, criterion: Optional[torch.nn.Module], runner_config: dict,
                                       batch_size: int, num_epochs: int, num_iterations_per_epoch: int, device: torch.device,
                                       torch_compile_options: TorchCompileOptions):
    optimization_config = runner_config['optimization']

    max_grad_norm = None
    if 'max_grad_norm' in optimization_config:
        max_grad_norm = optimization_config['max_grad_norm']
        print(f'optimization: max_grad_norm: {max_grad_norm}')
    grad_accumulation_steps = 1
    if 'grad_accumulation_steps' in optimization_config:
        grad_accumulation_steps = optimization_config['grad_accumulation_steps']
        print(f'optimization: grad_accumulation_steps: {grad_accumulation_steps}')
    assert num_iterations_per_epoch % grad_accumulation_steps == 0, \
        f'num_iterations_per_epoch ({num_iterations_per_epoch}) should be divisible by grad_accumulation_steps ({grad_accumulation_steps})'
    optimizer, lr, weight_decay, optimizer_logging_helper, is_apex_optimizer, clip_max_grad_norm_fused_by_optimizer = (
        build_optimizer(model, criterion, optimization_config['optimizer'], batch_size, grad_accumulation_steps, device, max_grad_norm))
    if clip_max_grad_norm_fused_by_optimizer:
        max_grad_norm = None

    lr_scheduler_per_iteration, lr_scheduler_per_epoch, lr_warmup_steps = build_lr_scheduler(optimization_config, optimizer, lr, num_epochs, num_iterations_per_epoch, grad_accumulation_steps)
    weight_decay_scheduler_per_iteration, weight_decay_scheduler_per_epoch = build_weight_decay_scheduler(optimization_config, optimizer, weight_decay, num_epochs, num_iterations_per_epoch)

    amp_parameter_updater, amp_auto_cast_fn = build_parameter_updater(optimization_config['auto_mixed_precision'], max_grad_norm, device, torch_compile_options.compiled_autograd)

    zero_grad_set_to_none = optimization_config.get('zero_grad_set_to_none', False)

    autograd_detect_anomaly_config = optimization_config.get('autograd_detect_anomaly', None)
    autograd_detect_anomaly_fn = nullcontext
    if autograd_detect_anomaly_config is not None and autograd_detect_anomaly_config['enabled']:
        detect_grad_nan = autograd_detect_anomaly_config.get('detect_grad_nan', True)
        autograd_detect_anomaly_fn = partial(torch.autograd.detect_anomaly, check_nan=detect_grad_nan)
        print(f'optimization: torch.autograd.detect_anomaly is enabled. check_nan: {detect_grad_nan}')

    return OptimizationModulesAndOptions(optimizer, optimizer_logging_helper, is_apex_optimizer,
                                         lr_scheduler_per_iteration, lr_scheduler_per_epoch,
                                         weight_decay_scheduler_per_iteration, weight_decay_scheduler_per_epoch,
                                         amp_parameter_updater, amp_auto_cast_fn,
                                         autograd_detect_anomaly_fn,
                                         grad_accumulation_steps, zero_grad_set_to_none,
                                         batch_size)
