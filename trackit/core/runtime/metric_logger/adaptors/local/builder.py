from typing import Optional
from trackit.core.runtime.build_context import BuildContext
from . import LocalMetricLoggerWrapper
from ... import MetricLogger


def construct_local_metric_logger(logger_dispatcher: MetricLogger, logging_config: Optional[dict], build_context: BuildContext):
    interval = 1
    prefix = None
    header = ''
    auto_summary = False

    if logging_config is not None:
        if 'local' in logging_config and 'interval' in logging_config['local']:
            interval = logging_config['local']['interval']
        elif 'interval' in logging_config:
            interval = logging_config['interval']

        if 'metric_prefix' in logging_config:
            prefix = logging_config['metric_prefix']

        if 'local' in logging_config:
            local_logger_config = logging_config['local']
            auto_summary = local_logger_config.get('auto_summary', auto_summary)
            if 'header' in local_logger_config:
                header = local_logger_config['header']

    metric_logger = LocalMetricLoggerWrapper(interval, prefix, auto_summary, header, build_context.device.type == 'cuda')
    build_context.services.event.register_on_epoch_begin_event_listener(lambda epoch, is_train: metric_logger.on_epoch_begin(epoch))
    build_context.services.event.register_on_epoch_end_event_listener(lambda epoch, is_train: metric_logger.on_epoch_end(epoch))
    logger_dispatcher.register_logger('local', metric_logger)
