import torch.optim


def build_lr_scheduler(optimization_config: dict, optimizer: torch.optim.Optimizer, lr: float, num_epochs: int, num_iterations_per_epoch: int, grad_accumulation_steps: int):
    if 'lr_scheduler' not in optimization_config:
        return None, None, 0
    lr_scheduler_config = optimization_config['lr_scheduler']
    if lr_scheduler_config['type'] == 'timm':
        from .timm_scheduler.builder import build_timm_lr_scheduler
        return build_timm_lr_scheduler(lr_scheduler_config, optimizer, lr, num_epochs, num_iterations_per_epoch, grad_accumulation_steps)
    else:
        raise NotImplementedError()
