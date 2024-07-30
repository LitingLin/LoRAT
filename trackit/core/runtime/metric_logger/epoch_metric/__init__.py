from typing import Dict, Optional


class EpochMetrics:
    def __init__(self):
        self.epoch_metrics: Dict[int, Dict[str, float]] = {}

    def reset(self):
        self.epoch_metrics.clear()

    def update(self, epoch: int, metrics: Dict[str, float]):
        if epoch not in self.epoch_metrics:
            self.epoch_metrics[epoch] = {}
        self.epoch_metrics[epoch].update(metrics)

    def get(self, epoch: int) -> Dict[str, float]:
        if epoch not in self.epoch_metrics:
            return {}
        return self.epoch_metrics[epoch]


_epoch_metrics: Optional[EpochMetrics] = None


def enable_epoch_metrics(epoch_metrics: EpochMetrics):
    global _epoch_metrics
    _epoch_metrics = epoch_metrics


def disable_epoch_metrics():
    global _epoch_metrics
    _epoch_metrics = None


def get_current_epoch_metrics():
    return _epoch_metrics
