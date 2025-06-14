import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional
from .optimization.parameter_updater.parameter_updater import ParameterUpdater_WithAMPSupport
import os.path
from trackit.miscellanies.torch.distributed import get_rank


def do_loss_nan_fault_dump(model: nn.Module,
                           optimizer: Optimizer,
                           lr_scheduler_per_iter: Optional[_LRScheduler],
                           lr_scheduler_per_epoch: Optional[_LRScheduler],
                           parameter_updater: ParameterUpdater_WithAMPSupport,
                           inputs, outputs,
                           loss_metrics: Dict[str, float],
                           output_path: str, task_name: str, epoch: int, iteration_index: int):
    save_path = os.path.join(output_path, f'nanfault-rank-{get_rank()}-{task_name}-e{epoch:03d}-iter-{iteration_index}.pth')
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'inputs': inputs,
        'outputs': outputs,
        'loss': loss_metrics
    }
    if lr_scheduler_per_iter is not None:
        state_dict['lr_scheduler'] = lr_scheduler_per_iter.state_dict()
    if lr_scheduler_per_epoch is not None:
        state_dict['lr_scheduler_per_epoch'] = lr_scheduler_per_epoch.state_dict()
    state_dict['amp_param_updater'] = parameter_updater.state_dict()
    torch.save(state_dict, save_path)
