import math

from typing import Optional
import torch
import torch.nn as nn
from .per_parameter_options.apply import parse_optimizer_per_params_config
from trackit.miscellanies.torch.distributed import get_world_size
from .logging_helper import OptimizerLoggingHelper


def build_optimizer(model: nn.Module, criterion: Optional[nn.Module],
                    optimizer_config: dict, batch_size: int, grad_accumulation_steps: int,
                    device: torch.device, max_grad_norm: Optional[float], zero_grad_set_to_none: bool = False):
    clip_max_grad_norm_fused_by_optimizer = False
    is_apex_optimizer = False
    optimizer_param_groups = parse_optimizer_per_params_config(model, criterion, optimizer_config)
    lr = optimizer_config['lr']
    if 'lr_auto_scaling' in optimizer_config:
        lr_auto_scaling_config = optimizer_config['lr_auto_scaling']
        lr_auto_scaling_reference_batch_size = lr_auto_scaling_config['reference_batch_size']
        world_size = get_world_size()
        effective_batch_size = batch_size * world_size * grad_accumulation_steps
        if effective_batch_size != lr_auto_scaling_reference_batch_size:
            if lr_auto_scaling_config['type'] == 'linear':
                lr_scaling_ratio = (batch_size * get_world_size()) / lr_auto_scaling_reference_batch_size
            elif lr_auto_scaling_config['type'] == 'sqrt':
                lr_scaling_ratio = math.sqrt(batch_size * get_world_size() / lr_auto_scaling_reference_batch_size)
            else:
                raise NotImplementedError(f'Unknown lr_auto_scaling type: {lr_auto_scaling_config["type"]}')
            lr *= lr_scaling_ratio
            print(f'lr scaled by {lr_scaling_ratio} to {lr}, rule type: {lr_auto_scaling_config["type"]}')
    weight_decay = optimizer_config['weight_decay']

    optimizer_parameters = {'lr': lr, 'weight_decay': weight_decay}

    if optimizer_config['type'] == 'SGD':
        if 'momentum' in optimizer_config:
            optimizer_parameters['momentum'] = optimizer_config['momentum']
        if 'nesterov' in optimizer_config:
            optimizer_parameters['nesterov'] = optimizer_config['nesterov']
        optimizer_cls = torch.optim.SGD
    elif optimizer_config['type'] == 'AdamW':
        if 'betas' in optimizer_config:
            optimizer_parameters['betas'] = optimizer_config['betas']
        if 'eps' in optimizer_config:
            optimizer_parameters['eps'] = optimizer_config['eps']
        if 'fused' in optimizer_config and int(torch.__version__.split('.')[0]) > 1:
            optimizer_parameters['fused'] = optimizer_config['fused']
            if optimizer_parameters['fused'] and device.type != 'cuda':
                print('fused AdamW only support CUDA, disabled.')
                optimizer_parameters['fused'] = False
            if optimizer_parameters['fused'] and torch.__version__.startswith('2.0.0'):
                print('workaround: fused AdamW may cause NaN in Pytorch 2.0.0, disabled.')
                optimizer_parameters['fused'] = False
        optimizer_cls = torch.optim.AdamW
    elif optimizer_config['type'] == 'Lamb':
        use_apex = optimizer_config['use_apex']
        if device.type != 'cuda' and use_apex:
            print('APEX FusedLAMB optimizer only support GPU training, use timm LAMB instead.')
            use_apex = False
        if use_apex:
            from apex.optimizers import FusedLAMB as Lamb
            is_apex_optimizer = True
        else:
            from timm.optim import Lamb
        if 'betas' in optimizer_config:
            optimizer_parameters['betas'] = optimizer_config['betas']
        if 'eps' in optimizer_config:
            optimizer_parameters['eps'] = optimizer_config['eps']
        if max_grad_norm is not None:
            optimizer_parameters['max_grad_norm'] = max_grad_norm
        if use_apex:
            optimizer_parameters['set_grad_none'] = zero_grad_set_to_none
        clip_max_grad_norm_fused_by_optimizer = True
        optimizer_cls = Lamb
    elif optimizer_config['type'] == 'lion':
        from .lion import Lion
        if 'betas' in optimizer_config:
            optimizer_parameters['betas'] = optimizer_config['betas']
        optimizer_cls = Lion
    elif optimizer_config['type'] == 'prodigy':
        from .prodigy import Prodigy

        if 'betas' in optimizer_config:
            optimizer_parameters['betas'] = optimizer_config['betas']

        if 'parameters' in optimizer_config:
            optimizer_parameters.update(optimizer_config['parameters'])
        optimizer_cls = Prodigy
    else:
        raise NotImplementedError(f'Unknown lr_scheduler {optimizer_config["type"]}')

    print(f'optimizer: {optimizer_cls.__name__}: ' + ', '.join([f'{k}: {v}' for k, v in optimizer_parameters.items()]))
    if 'ZeRO' in optimizer_config and optimizer_config['ZeRO']['enabled']:
        zero_config = optimizer_config['ZeRO']
        parameters_as_bucket_view = zero_config.get('parameters_as_bucket_view', True)
        from torch.distributed.optim import ZeroRedundancyOptimizer
        print('optimizer: ZeroRedundancyOptimizer is enabled, parameters_as_bucket_view: '+ str(parameters_as_bucket_view))
        optimizer = ZeroRedundancyOptimizer(optimizer_param_groups, optimizer_cls,
                                            parameters_as_bucket_view=parameters_as_bucket_view, **optimizer_parameters)
    else:
        optimizer = optimizer_cls(optimizer_param_groups, **optimizer_parameters)

    return (optimizer, lr, weight_decay, OptimizerLoggingHelper(optimizer, lr, weight_decay),
            is_apex_optimizer, clip_max_grad_norm_fused_by_optimizer)
