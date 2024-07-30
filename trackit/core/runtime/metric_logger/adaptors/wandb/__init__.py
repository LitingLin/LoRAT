import wandb
import copy
from ...interface import MetricLoggerInterface


class WandbLogger(MetricLoggerInterface):
    def __init__(self, wandb_instance: wandb.wandb_sdk.wandb_run.Run, logging_frequency: int, prefix=None,
                 with_epoch=False, log_last=True):
        self.instance = wandb_instance
        self.logging_frequency = logging_frequency
        self.prefix = prefix
        self.with_epoch = with_epoch
        self.log_last = log_last

    def log_summary(self, meters):
        if self.prefix is not None:
            meters = {self.prefix + k: v for k, v in meters.items()}
        self.instance.summary.update(meters)

    def on_epoch_begin(self, epoch: int):
        self.logging_step = 0
        self.epoch = epoch
        self.last = None

    def on_epoch_end(self):
        if self.last is not None:
            self.instance.log(self.last[0], step=self.last[1])

        self.last = None

    def log(self, meters, force, step):
        if self.prefix is not None:
            meters = {self.prefix + k: v for k, v in meters.items()}
            if self.with_epoch:
                meters['epoch'] = self.epoch
        else:
            if self.with_epoch:
                meters = copy.copy(meters)
                meters['epoch'] = self.epoch

        if self.logging_step % self.logging_frequency == 0 or force:
            self.instance.log(meters, step=step)
            self.last = None
        else:
            if self.log_last:
                self.last = (meters, step)

    def commit(self):
        self.logging_step += 1

    def define_metrics(self, metric_definitions):
        for metric_definition in metric_definitions:
            if self.prefix is not None:
                metric_definition = copy.copy(metric_definition)
                metric_definition['name'] = metric_definition['name'] + self.prefix

            self.instance.define_metric(**metric_definition)


class WandbEpochSummaryLogger(MetricLoggerInterface):
    def __init__(self, wandb_instance: wandb.wandb_sdk.wandb_run.Run, summary_method, prefix=None, with_epoch=False):
        assert summary_method == 'mean'
        self.summary_method = summary_method
        self.instance = wandb_instance
        self.prefix = prefix
        self.with_epoch = with_epoch

    def log_summary(self, meters):
        if self.prefix is not None:
            meters = {self.prefix + k: v for k, v in meters.items()}
        self.instance.summary.update(meters)

    def on_epoch_begin(self, epoch: int):
        self.epoch = epoch
        self.metrics = {}
        self.step = None

    def on_epoch_end(self, epoch: int):
        if len(self.metrics) > 0:
            epoch_metrics = {}
            for metric_name, (metric_total, metric_count) in self.metrics.items():
                epoch_metrics[metric_name] = metric_total / metric_count
            if self.with_epoch:
                epoch_metrics['epoch'] = epoch
            self.instance.log(epoch_metrics, step=self.step)
        self.step = None

    def log(self,
            metrics,
            force,  # ignored
            step):
        if self.prefix is not None:
            metrics = {self.prefix + k: v for k, v in metrics.items()}

        for metric_name, metric_value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = [metric_value, 1]
            else:
                epoch_metric_statistic = self.metrics[metric_name]
                epoch_metric_statistic[0] += metric_value
                epoch_metric_statistic[1] += 1

        self.step = step

    def define_metrics(self, metric_definitions):
        for metric_definition in metric_definitions:
            if self.prefix is not None:
                metric_definition = copy.copy(metric_definition)
                metric_definition['name'] = metric_definition['name'] + self.prefix

            self.instance.define_metric(**metric_definition)
