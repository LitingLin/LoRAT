import torch.optim


class OptimizerLoggingHelper:
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, initial_weight_decay: float):
        self.optimizer = optimizer
        self._lr_monitoring_index = _find_valid_param_group(optimizer, 'lr')
        self._weight_decay_monitoring_index = _find_valid_param_group(optimizer, 'weight_decay')

        if self._lr_monitoring_index >= 0 and 'lr' in optimizer.param_groups[self._lr_monitoring_index]:
            self._lr_scaling_ratio = initial_lr / optimizer.param_groups[0]['lr']
        else:
            self._lr_scaling_ratio = 1
        if self._weight_decay_monitoring_index >= 0 and 'weight_decay' in optimizer.param_groups[self._weight_decay_monitoring_index]:
            self._weight_decay_scaling_ratio = initial_weight_decay / optimizer.param_groups[self._weight_decay_monitoring_index]['weight_decay']
        else:
            self._weight_decay_scaling_ratio = 1

    def get_lr(self):
        if self._lr_monitoring_index < 0:
            return 0
        return self.optimizer.param_groups[self._lr_monitoring_index]['lr'] * self._lr_scaling_ratio

    def get_weight_decay(self):
        if self._weight_decay_monitoring_index < 0:
            return 0
        return self.optimizer.param_groups[self._weight_decay_monitoring_index]['weight_decay'] * self._weight_decay_scaling_ratio

def _find_valid_param_group(optimizer: torch.optim.Optimizer, name: str):
    for i, param_group in enumerate(optimizer.param_groups):
        if name not in param_group:
            return i

    for i, param_group in enumerate(optimizer.param_groups):
        if name in param_group and param_group[name] > 0:
            return i
    return -1
