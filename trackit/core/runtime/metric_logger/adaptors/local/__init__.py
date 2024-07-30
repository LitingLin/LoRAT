from typing import Optional, Iterable
from trackit.miscellanies.torch.metric_logger import LocalMetricLogger, ProgressTrackerInterface
from tabulate import tabulate
from ...interface import MetricLoggerInterface
from ...epoch_metric import get_current_epoch_metrics


class LocalMetricLoggerWrapper(MetricLoggerInterface):
    def __init__(self, print_freq: int, prefix: Optional[str], epoch_average_as_summary: bool, header: str, monitor_cuda_device_memory: bool=False):
        self.print_freq = print_freq
        self.prefix = prefix
        self.epoch_average_as_summary = epoch_average_as_summary
        self.header = header
        self.no_name_prefix_meters = set()
        self.logger = LocalMetricLogger(delimiter=' ', print_freq=print_freq)
        # self.logger.enable_monitoring_cpu_percent()
        self.logger.enable_monitoring_system_total_resident_set_size()
        if monitor_cuda_device_memory:
            self.logger.enable_monitoring_cuda_device_memory_allocated()
        self.summary = {}

    def on_epoch_begin(self, epoch: int) -> None:
        self.epoch_header = self.header.format(epoch=epoch)

    def on_epoch_end(self, epoch: int) -> None:
        summary = {}
        if self.epoch_average_as_summary:
            for name, meter in self.logger.meters.items():
                summary[name] = meter.global_avg
        summary.update(self.summary)
        if len(summary) > 0:
            epoch_metrics = get_current_epoch_metrics()
            if epoch_metrics is not None:
                epoch_metrics.update(epoch=epoch, metrics=self.summary)
            print(f'{self.epoch_header} summary metrics:\n' + tabulate(summary.items(), headers=('metric', 'value'), floatfmt=".4f"), flush=True)
        self.reset()
        self.epoch_header = None

    def log(self, meters,
            force: bool = False,  # ignore
            step: int = 0   # ignore
            ) -> None:
        if self.prefix is not None:
            meters = {self.prefix + k if k not in self.no_name_prefix_meters else k: v for k, v in meters.items()}
        self.logger.update(**meters)

    def log_summary(self, meters) -> None:
        if self.prefix is not None:
            meters = {self.prefix + k if k not in self.no_name_prefix_meters else k: v for k, v in meters.items()}
        self.summary.update(meters)

    def log_every(self, iterable: Iterable):
        return self.logger.log_every(iterable, self.epoch_header)

    def set_metric_format(self, name: str, window_size: int = 20, format: str = "{median:.4f} ({global_avg:.4f})", no_prefix=False) -> None:
        if no_prefix or self.prefix is None:
            self.logger.add_meter(name, window_size, format)
            if no_prefix not in self.no_name_prefix_meters:
                self.no_name_prefix_meters.add(name)
        else:
            self.logger.add_meter(self.prefix + name, window_size, format)
            if name in self.no_name_prefix_meters:
                self.no_name_prefix_meters.remove(name)

    def set_custom_progress_tracker(self, progress_tracker: Optional[ProgressTrackerInterface]) -> None:
        self.logger.set_custom_progress_tracker(progress_tracker)

    def reset(self):
        self.logger.reset()
        self.no_name_prefix_meters.clear()
        self.summary.clear()
