
class MetricLoggerInterface:
    def log(self, meters, force: bool, step: int) -> None:
        raise NotImplementedError

    def log_summary(self, meters) -> None:
        raise NotImplementedError

    def commit(self) -> None:
        pass

