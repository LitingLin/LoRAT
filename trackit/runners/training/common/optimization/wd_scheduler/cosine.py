""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise, k-decay.

Hacked together by / Copyright 2021 Ross Wightman

Modified to work with weight decay scheduling
"""
import logging
import math
import torch

from timm.scheduler.scheduler import Scheduler


_logger = logging.getLogger(__name__)


class CosineWDScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            wd_end: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_wd_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="weight_decay",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert wd_end >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.wd_end = wd_end
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_wd_init = warmup_wd_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_wd_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_wd_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            wds = [self.warmup_wd_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            wd_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                wds = [
                    self.wd_end + 0.5 * (wd_max - self.wd_end) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for wd_max in wd_max_values
                ]
            else:
                wds = [self.wd_end for _ in self.base_values]

        return wds

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            if 'weight_decay_scale' in param_group:
                param_group[self.param_group_field] = value * param_group['weight_decay_scale']
            else:
                param_group[self.param_group_field] = value
