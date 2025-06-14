import torch.optim


class OptimizerLoggingHelper:
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, initial_weight_decay: float):
        self.optimizer = optimizer
        if 'lr' in optimizer.param_groups[0]:
            self._lr_scaling_ratio = initial_lr / optimizer.param_groups[0]['lr']
        else:
            self._lr_scaling_ratio = 1.0
        if 'weight_decay' in optimizer.param_groups[0]:
            self._weight_decay_scaling_ratio = initial_weight_decay / optimizer.param_groups[0]['weight_decay']
        else:
            self._weight_decay_scaling_ratio = 1.0

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr'] * self._lr_scaling_ratio

    def get_weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay'] * self._weight_decay_scaling_ratio
