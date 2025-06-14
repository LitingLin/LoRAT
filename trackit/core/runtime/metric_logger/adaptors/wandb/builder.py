from . import WandbLogger, WandbEpochSummaryLogger
from trackit.core.runtime.build_context import BuildContext
from typing import Optional
from ... import MetricLogger


def construct_wandb_metric_logger(logger_dispatcher: MetricLogger, logging_config: Optional[dict], build_context: BuildContext):
    wandb_instance = build_context.wandb_instance
    assert wandb_instance is not None
    if logging_config is None:
        logger = WandbLogger(wandb_instance, 1)
        build_context.services.event.register_on_epoch_begin_event_listener(
            lambda epoch, is_train: logger.on_epoch_begin(epoch))
        build_context.services.event.register_on_epoch_end_event_listener(
            lambda epoch, is_train: logger.on_epoch_end())
        logger_dispatcher.register_logger('wandb', logger, 'external')
    enable_per_iteration_logging = True
    enable_per_epoch_logging = False
    per_iteration_logging_prefix = None
    per_epoch_logging_prefix = None
    with_epoch = False
    logging_interval = 1
    per_epoch_logging_summary_method = 'mean'

    if 'wandb' in logging_config and 'interval' in logging_config['wandb']:
        logging_interval = logging_config['wandb']['interval']
    elif 'interval' in logging_config:
        logging_interval = logging_config['interval']

    if 'metric_prefix' in logging_config:
        per_iteration_logging_prefix = logging_config['metric_prefix']
        per_epoch_logging_prefix = logging_config['metric_prefix']
    if 'wandb' in logging_config:
        wandb_config = logging_config['wandb']
        if 'with_epoch' in wandb_config:
            with_epoch = wandb_config['with_epoch']

        if 'per_iteration_logging' in wandb_config:
            per_iteration_logging_config = wandb_config['per_iteration_logging']
            enable_per_iteration_logging = per_iteration_logging_config['enabled']
            if 'prefix' in per_iteration_logging_config:
                per_iteration_logging_prefix = per_iteration_logging_prefix + per_iteration_logging_config['prefix'] if per_iteration_logging_prefix is not None else per_iteration_logging_config['prefix']

        if 'per_epoch_logging' in wandb_config:
            per_epoch_logging_config = wandb_config['per_epoch_logging']
            enable_per_epoch_logging = per_epoch_logging_config['enabled']
            if 'prefix' in per_epoch_logging_config:
                per_epoch_logging_prefix = per_epoch_logging_prefix + per_epoch_logging_config['prefix'] if per_epoch_logging_prefix is not None else per_epoch_logging_config['prefix']

            if 'summary_method' in per_epoch_logging_config:
                per_epoch_logging_summary_method = per_epoch_logging_config['summary_method']

    if enable_per_iteration_logging:
        logger = WandbLogger(wandb_instance, logging_interval, per_iteration_logging_prefix, with_epoch)
        build_context.services.event.register_on_epoch_begin_event_listener(
            lambda epoch, is_train: logger.on_epoch_begin(epoch))
        build_context.services.event.register_on_epoch_end_event_listener(
            lambda epoch, is_train: logger.on_epoch_end())
        logger_dispatcher.register_logger('wandb', logger, 'external')
    if enable_per_epoch_logging:
        logger = WandbEpochSummaryLogger(wandb_instance, per_epoch_logging_summary_method, per_epoch_logging_prefix, with_epoch)
        build_context.services.event.register_on_epoch_begin_event_listener(
            lambda epoch, is_train: logger.on_epoch_begin(epoch))
        build_context.services.event.register_on_epoch_end_event_listener(
            lambda epoch, is_train: logger.on_epoch_end(epoch))
        logger_dispatcher.register_logger('wandb_epoch_summary', logger, 'external')
