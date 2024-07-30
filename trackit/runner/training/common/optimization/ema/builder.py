import torch.nn as nn
from . import EMAModule
from trackit.miscellanies.torch.distributed import get_world_size


def build_ema(optimization_config: dict, model: nn.Module, batch_size: int, num_epochs: int, lr_warmup_steps: int, grad_accumulation_steps: int):
    if 'ema' not in optimization_config or not optimization_config['ema']['enabled']:
        return None

    ema_config = optimization_config['ema']
    batch_size = batch_size * get_world_size()
    ema_steps = ema_config['steps']
    ema_steps *= grad_accumulation_steps
    lr_warmup_steps = lr_warmup_steps // grad_accumulation_steps
    return EMAModule(model, ema_config['decay'], ema_steps, batch_size, num_epochs, lr_warmup_steps)
