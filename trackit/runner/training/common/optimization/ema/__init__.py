import copy
from typing import Optional

import torch
import torch.nn as nn
from .exponential_moving_average import ExponentialMovingAverage


class EMAModule:
    def __init__(self, model: nn.Module, ema_decay: float, ema_steps: int, batch_size: int, num_epochs: int, lr_warmup_steps: int, device: torch.device = torch.device('cpu')):
        adjust = batch_size * ema_steps / num_epochs
        alpha = 1.0 - ema_decay
        alpha = min(1.0, alpha * adjust)
        self.ema_steps = ema_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.ema = ExponentialMovingAverage(model, 1.0 - alpha, device)
        self.step = 0
        self.global_step = 0

    def on_epoch_begin(self):
        self.step = 0

    def update_parameters(self, model: nn.Module):
        if self.step % self.ema_steps == 0:
            self.ema.update_parameters(model)
            if self.global_step < self.lr_warmup_steps:
                # Reset ema buffer to keep copying weights during warmup period
                self.ema.n_averaged.fill_(0)
        self.step += 1
        self.global_step += 1

    def get_model(self):
        return self.ema.module

    def state_dict(self):
        return self.ema.n_averaged.cpu().item(), self.step, self.global_step

    def load_state_dict(self, state):
        self.ema.n_averaged.fill_(state[0])
        self.step = state[1]
        self.global_step = state[2]
