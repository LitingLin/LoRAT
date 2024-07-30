from . import MetricLogger
from trackit.core.runtime.build_context import BuildContext
from .adaptors.local.builder import construct_local_metric_logger
from .adaptors.wandb.builder import construct_wandb_metric_logger


def build_metric_logger(logging_config, build_context: BuildContext) -> MetricLogger:
    logger_dispatcher = MetricLogger()
    construct_local_metric_logger(logger_dispatcher, logging_config, build_context)
    if build_context.wandb_instance is not None:
        construct_wandb_metric_logger(logger_dispatcher, logging_config, build_context)
    return logger_dispatcher
