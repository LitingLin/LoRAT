import torch.optim


def build_timm_lr_scheduler(lr_scheduler_config: dict, optimizer: torch.optim.Optimizer, lr: float, num_epochs: int,
                            num_iterations_per_epoch: int, grad_accumulation_steps: int, wrap_with_torch_lr_scheduler: bool = False):
    sched_type = lr_scheduler_config['sched']
    per_iteration = lr_scheduler_config.get('per_iteration', True)
    if 'override' in lr_scheduler_config:
        num_epochs = lr_scheduler_config['override']['num_epochs']
        print(f'lr_scheduler: assuming num_epochs={num_epochs} from override rule')
    config = {'warmup_prefix': False, 'warmup_epochs': 0, 'warmup_lr_mul': 1.}

    if 'parameters' in lr_scheduler_config:
        config.update(lr_scheduler_config['parameters'])

    warmup_lr = config['warmup_lr'] if 'warmup_lr' in config else lr * config['warmup_lr_mul']

    num_updates_per_epoch = num_iterations_per_epoch // grad_accumulation_steps

    warmup_logging_items = {}
    if 'warmup_steps' in config:
        assert per_iteration
        warmup_t = config['warmup_steps']
        warmup_logging_items['warmup_steps'] = warmup_t
    else:
        warmup_t = config['warmup_epochs']
        warmup_logging_items['warmup_epochs'] = warmup_t
        if per_iteration:
            warmup_t = int(num_updates_per_epoch * warmup_t)

    warmup_logging_items.update({'warmup_lr': warmup_lr, 'warmup_prefix': config['warmup_prefix']})

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

        scheduler_logging_items = {}
        if 'decay_epochs' in config:
            decay_t = config['decay_epochs']
            scheduler_logging_items['decay_epochs'] = decay_t
            if per_iteration:
                decay_t = int(num_updates_per_epoch * decay_t)
        elif 'decay_steps' in config:
            decay_t = config['decay_steps']
            scheduler_logging_items['decay_steps'] = decay_t
            assert per_iteration
        else:
            raise ValueError('Either decay_epochs or decay_steps must be provided in config')

        decay_rate = config.get('decay_rate', 1.)
        scheduler_logging_items.update({'decay_rate': decay_rate})

        lr_scheduler = StepLRScheduler(optimizer,
                                       decay_t=decay_t,
                                       decay_rate=decay_rate,
                                       warmup_prefix=config['warmup_prefix'],
                                       warmup_t=warmup_t,
                                       warmup_lr_init=warmup_lr,
                                       t_in_epochs=not per_iteration)
    elif sched_type == 'multi_step':
        from timm.scheduler.multistep_lr import MultiStepLRScheduler

        scheduler_logging_items = {}
        if 'decay_milestone_epochs' in config:
            decay_t = config['decay_milestone_epochs']
            scheduler_logging_items['decay_milestone_epochs'] = decay_t
            if per_iteration:
                decay_t = [int(t * num_updates_per_epoch) for t in decay_t]
        elif 'decay_milestone_steps' in config:
            decay_t = config['decay_milestone_steps']
            scheduler_logging_items['decay_milestone_steps'] = decay_t
            assert per_iteration
        else:
            raise ValueError('Either decay_milestone_epochs or decay_milestone_steps must be provided in config')

        decay_rate = config.get('decay_rate', 1.)
        scheduler_logging_items.update({'decay_rate': decay_rate})

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

    lr_warmup_steps = warmup_t
    if per_iteration:
        lr_warmup_steps = lr_warmup_steps * grad_accumulation_steps

    if wrap_with_torch_lr_scheduler:
        from .wrapper import TimmLRSchedulerWrapper
        lr_scheduler = TimmLRSchedulerWrapper(lr_scheduler)

    return lr_scheduler if per_iteration else None, lr_scheduler if not per_iteration else None, lr_warmup_steps
