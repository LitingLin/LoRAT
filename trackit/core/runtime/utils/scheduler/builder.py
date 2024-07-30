from ...build_context import BuildContext


def build_scheduler(scheduler_config: dict, build_context: BuildContext, name: str, total_iterations: int):
    if scheduler_config['type'] == 'constant':
        from .constant import ConstantScheduler

        weight_scheduler = ConstantScheduler(scheduler_config['value'])
    elif scheduler_config['type'] == 'linear':
        from .linear import LinearScheduler

        weight_scheduler = LinearScheduler(scheduler_config['initial_value'],
                                           scheduler_config['ultimate_value'],
                                           int(round(scheduler_config['initial_milestone'] * total_iterations)),
                                           int(round(scheduler_config['ultimate_milestone'] * total_iterations)),
                                           scheduler_config['per_iteration'])
    else:
        raise NotImplementedError(f'Unknown weight scheduler type {scheduler_config["type"]}')

    build_context.services.event.register_on_iteration_begin_event_listener(weight_scheduler.on_iteration_begin)
    build_context.services.event.register_on_iteration_end_event_listener(weight_scheduler.on_iteration_end)
    build_context.services.event.register_on_epoch_begin_event_listener(weight_scheduler.on_epoch_begin)
    build_context.services.event.register_on_epoch_end_event_listener(weight_scheduler.on_epoch_end)

    build_context.services.checkpoint.register(name, weight_scheduler.get_state, weight_scheduler.set_state)

    return weight_scheduler
