from typing import Dict, List, Optional, Union, Mapping
from .interface import MetricLoggerInterface
from .adaptors.local import LocalMetricLoggerWrapper


class MetricLogger:
    def __init__(self):
        self.loggers: Dict[str, MetricLoggerInterface] = {}
        self.logger_groups: Dict[str, List[MetricLoggerInterface]] = {}
        self.step: Optional[int] = None

    def register_logger(self, name: str, logger: MetricLoggerInterface, group_name: Optional[str] = None) -> None:
        assert name not in ('all', 'meters')
        assert name not in self.loggers
        self.loggers[name] = logger
        if group_name is not None:
            assert group_name not in self.loggers
            if group_name not in self.logger_groups:
                self.logger_groups[group_name] = []
            self.logger_groups[group_name].append(logger)

    def get_logger(self, name: str) -> MetricLoggerInterface:
        return self.loggers[name]

    def has_logger(self, name: str) -> bool:
        return name in self.loggers

    def log(self, all: Optional[Mapping[str, Union[float, int]]]=None, auto_increment: bool=False, force: bool=False, **kwargs) -> None:
        step = self.step
        if all is not None:
            for logger in self.loggers.values():
                logger.log(all, force=force, step=step)

        for logger_name in kwargs.keys():
            if logger_name in self.loggers:
                self.loggers[logger_name].log(kwargs[logger_name], force=force, step=step)
            if logger_name in self.logger_groups:
                for logger in self.logger_groups[logger_name]:
                    logger.log(kwargs[logger_name], force=force, step=step)

        if auto_increment:
            self.step += 1

    def log_summary(self, meters=None, **kwargs) -> None:
        if meters is not None:
            for logger in self.loggers.values():
                logger.log_summary(meters)

        for logger_name in kwargs.keys():
            if logger_name in self.loggers:
                self.loggers[logger_name].log_summary(kwargs[logger_name])
            if logger_name in self.logger_groups:
                for logger in self.logger_groups[logger_name]:
                    logger.log_summary(kwargs[logger_name])

    def commit(self) -> None:
        for logger in self.loggers.values():
            logger.commit()

    def set_step(self, step: int) -> None:
        self.step = step


_logger: Optional[MetricLogger] = None
_local_logger: Optional[LocalMetricLoggerWrapper] = None


class enable_metric_logger:
    def __init__(self, logger: MetricLogger, local_logger: Optional[LocalMetricLoggerWrapper]):
        global _logger
        global _local_logger
        self._last_logger = _logger
        self._last_local_logger = _local_logger
        _logger = logger
        _local_logger = local_logger

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _logger
        global _local_logger
        _logger = self._last_logger
        _local_logger = self._last_local_logger


def disable_metric_logger():
    global _logger
    _logger = None


def get_current_metric_logger() -> Optional[MetricLogger]:
    return _logger


def get_current_local_metric_logger() -> Optional[LocalMetricLoggerWrapper]:
    return _local_logger
