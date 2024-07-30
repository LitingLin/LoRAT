import torch.optim


def build_timm_lr_scheduler(lr_scheduler_config: dict, optimizer: torch.optim.Optimizer, lr: float, num_epochs: int, num_iterations_per_epoch: int, grad_accumulation_steps: int):
    sched_type = lr_scheduler_config['sched']
    per_iteration = lr_scheduler_config.get('per_iteration', True)
    if 'override' in lr_scheduler_config:
        num_epochs = lr_scheduler_config['override']['num_epochs']
        print(f'lr_scheduler: assuming num_epochs={num_epochs} from override rule')
    config = {'warmup_prefix': False, 'warmup_epochs': 0,
              'warmup_lr_mul': 1.}

    if 'parameters' in lr_scheduler_config:
        config.update(lr_scheduler_config['parameters'])

    warmup_t = config['warmup_epochs']
    warmup_lr = config['warmup_lr'] if 'warmup_lr' in config else lr * config['warmup_lr_mul']

    warmup_logging_items = {'warmup_epochs': warmup_t, 'warmup_lr': warmup_lr, 'warmup_prefix': config['warmup_prefix']}

    num_updates_per_epoch = num_iterations_per_epoch // grad_accumulation_steps

    if per_iteration:
        warmup_t *= num_updates_per_epoch

    if sched_type == 'cosine':
        from timm.scheduler.cosine_lr import CosineLRScheduler

        lr_min = config.get('lr_min', 0.)
        cycle_mul = config.get('cycle_mul', 1)
        cycle_decay = config.get('cycle_decay', 1)
        cycle_limit = config.get('cycle_limit', 1)
        cooldown_epochs = config.get('cooldown_epochs', 0)
        scheduler_logging_items = {'lr_min': lr_min, 'cycle_mul': cycle_mul, 'cycle_decay': cycle_decay,
                                   'cycle_limit': cycle_limit, 'cooldown_epochs': cooldown_epochs}
        t_initial = num_epochs - cooldown_epochs
        if t_initial <= 0:
            t_initial = 1
        if per_iteration:
            t_initial *= num_updates_per_epoch
        lr_scheduler = CosineLRScheduler(optimizer, t_initial,
                                         lr_min=lr_min,
                                         cycle_mul=cycle_mul,
                                         cycle_decay=cycle_decay,
                                         cycle_limit=cycle_limit,
                                         warmup_prefix=config['warmup_prefix'],
                                         warmup_t=warmup_t,
                                         warmup_lr_init=warmup_lr,
                                         t_in_epochs=not per_iteration)
    elif sched_type == 'step':
        from timm.scheduler.step_lr import StepLRScheduler
        decay_t = config['decay_step']
        decay_rate = config.get('decay_rate', 1.)

        scheduler_logging_items = {'decay_step': decay_t, 'decay_rate': decay_rate}
        if per_iteration:
            decay_t *= num_updates_per_epoch

        lr_scheduler = StepLRScheduler(optimizer,
                                       decay_t=decay_t,
                                       decay_rate=decay_rate,
                                       warmup_prefix=config['warmup_prefix'],
                                       warmup_t=warmup_t,
                                       warmup_lr_init=warmup_lr,
                                       t_in_epochs=not per_iteration)
    elif sched_type == 'multi_step':
        from timm.scheduler.multistep_lr import MultiStepLRScheduler

        decay_t = config['decay_milestones']
        decay_rate = config.get('decay_rate', 1.)
        scheduler_logging_items = {'decay_milestones': decay_t, 'decay_rate': decay_rate}
        if per_iteration:
            decay_t = [t * num_updates_per_epoch for t in decay_t]

        lr_scheduler = MultiStepLRScheduler(optimizer,
                                            decay_t=decay_t,
                                            decay_rate=decay_rate,
                                            warmup_prefix=config['warmup_prefix'],
                                            warmup_t=warmup_t,
                                            warmup_lr_init=warmup_lr,
                                            t_in_epochs=not per_iteration)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler {sched_type}')

    print(f'lr_scheduler: {sched_type}: '
          + ', '.join([f'{k}:{v}' for k, v in scheduler_logging_items.items()]) + ', ' +
          ', '.join([f'{k}:{v}' for k, v in warmup_logging_items.items()]) + f', per_iteration:{per_iteration}')

    lr_warmup_steps = config['warmup_epochs']
    if per_iteration:
        lr_warmup_steps *= num_iterations_per_epoch

    return lr_scheduler if per_iteration else None, lr_scheduler if not per_iteration else None, lr_warmup_steps
