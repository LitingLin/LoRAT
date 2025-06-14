import torch
from .parameter_updater.parameter_updater import ParameterUpdater_WithAMPSupport
from dataclasses import dataclass
from typing import Optional, Union, Callable
from timm.scheduler.scheduler import Scheduler as timmScheduler
from .optimizer.logging_helper import OptimizerLoggingHelper


@dataclass(frozen=True)
class OptimizationModulesAndOptions:
    optimizer: torch.optim.Optimizer
    optimizer_logging_helper: OptimizerLoggingHelper
    is_apex_optimizer: bool
    lr_scheduler_per_iteration: Optional[Union[torch.optim.lr_scheduler._LRScheduler, timmScheduler]]
    lr_scheduler_per_epoch: Optional[Union[torch.optim.lr_scheduler._LRScheduler, timmScheduler]]
    weight_decay_scheduler_per_iteration: Optional[timmScheduler]
    weight_decay_scheduler_per_epoch: Optional[timmScheduler]
    parameter_updater: Optional[ParameterUpdater_WithAMPSupport]
    amp_auto_cast_fn: Callable
    autograd_detect_anomaly_fn: Callable
    grad_accumulation_steps: int
    zero_grad_set_to_none: bool
    train_batch_size: int
