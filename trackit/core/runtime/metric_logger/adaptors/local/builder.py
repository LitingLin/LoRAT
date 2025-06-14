from typing import Optional
from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.global_constant import get_global_constant
from . import LocalMetricLoggerWrapper
from ... import MetricLogger


def construct_local_metric_logger(logger_dispatcher: MetricLogger, logging_config: Optional[dict], build_context: BuildContext):
    interval = 1
    prefix = None
    header = ''
    auto_summary = False
    monitor_system_resources = get_global_constant('monitor_system_resources', default=True)
    monitor_system_health_sensors = get_global_constant('monitor_system_health_sensors', default=False)

    if logging_config is not None:
        if 'local' in logging_config and 'interval' in logging_config['local']:
            interval = logging_config['local']['interval']
        elif 'interval' in logging_config:
            interval = logging_config['interval']

        prefix = logging_config.get('metric_prefix', prefix)

        if 'local' in logging_config:
            local_logger_config = logging_config['local']
            auto_summary = local_logger_config.get('auto_summary', auto_summary)
            header = local_logger_config.get('header', header)
            monitor_system_resources = local_logger_config.get('monitor_system_resources',
                                                               monitor_system_resources)
            monitor_system_health_sensors = local_logger_config.get('monitor_system_health_sensors',
                                                                    monitor_system_health_sensors)

    monitor_cuda_device = build_context.device.type == 'cuda'
    monitor_mps_device = build_context.device.type == 'mps'
    metric_logger = LocalMetricLoggerWrapper(interval, prefix, auto_summary, header,
                                             monitor_system_resources, monitor_system_health_sensors,
                                             monitor_cuda_device, monitor_mps_device)
    build_context.services.event.register_on_epoch_begin_event_listener(
        lambda epoch, is_train: metric_logger.on_epoch_begin(epoch))
    build_context.services.event.register_on_epoch_end_event_listener(
        lambda epoch, is_train: metric_logger.on_epoch_end(epoch))
    logger_dispatcher.register_logger('local', metric_logger)
