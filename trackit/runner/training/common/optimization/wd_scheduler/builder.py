import torch.optim


def build_timm_weight_decay_scheduler(weight_decay_scheduler_config: dict,
                                      optimizer: torch.optim.Optimizer, weight_decay: float,
                                      num_epochs: int, num_iterations_per_epoch: int):
    sched_type = weight_decay_scheduler_config['sched']
    per_iteration = weight_decay_scheduler_config['per_iteration']
    if sched_type == 'cosine':
        from .cosine import CosineWDScheduler
        t_initial = num_epochs
        config = {'wd_end': 0., 'cycle_mul': 1, 'cycle_decay': 0.1,
                  'cycle_limit': 1, 'warmup_prefix': False, 'warmup_epochs': 0,
                  'warmup_wd_mul': 0.001}
        if 'parameters' in weight_decay_scheduler_config:
            config.update(weight_decay_scheduler_config['parameters'])
        warmup_t = config['warmup_epochs']
        warmup_wd = weight_decay * config['warmup_wd_mul']
        if per_iteration:
            t_initial *= num_iterations_per_epoch
            warmup_t *= num_iterations_per_epoch
        wd_scheduler = CosineWDScheduler(optimizer, t_initial,
                                         wd_end=config['wd_end'],
                                         cycle_mul=config['cycle_mul'],
                                         cycle_decay=config['cycle_decay'],
                                         cycle_limit=config['cycle_limit'],
                                         warmup_prefix=config['warmup_prefix'],
                                         warmup_t=warmup_t,
                                         warmup_wd_init=warmup_wd,
                                         t_in_epochs=not per_iteration)

        print(f'wd_scheduler: {sched_type}: '
              + ', '.join([f'{k}:{v}' for k, v in config.items()]) + f', per_iteration:{per_iteration}')
    else:
        raise NotImplementedError(f'Unknown weight_decay_scheduler {sched_type}')

    return wd_scheduler if per_iteration else None, wd_scheduler if not per_iteration else None


def build_weight_decay_scheduler(optimization_config: dict, optimizer: torch.optim.Optimizer,
                                 weight_decay: float,
                                 num_epochs: int, num_iterations_per_epoch: int):
    if 'weight_decay_scheduler' not in optimization_config:
        return None, None
    weight_decay_scheduler_config = optimization_config['weight_decay_scheduler']
    if weight_decay_scheduler_config['type'] == 'timm':
        return build_timm_weight_decay_scheduler(weight_decay_scheduler_config, optimizer,
                                                 weight_decay, num_epochs, num_iterations_per_epoch)
    else:
        raise NotImplementedError(f'Unknown weight_decay_scheduler {weight_decay_scheduler_config["type"]}')
