from typing import Any, Optional
from trackit.models import ModelManager
from trackit.data import HostDataPipeline


class Runner:
    def register_data_pipeline(self, task_name: Optional[str], data_pipeline: HostDataPipeline) -> None:
        raise NotImplementedError()

    def switch_task(self, task_name: Optional[str], is_train: bool) -> None:
        raise NotImplementedError()

    def run(self, data) -> None:
        raise NotImplementedError()

    def epoch_begin(self, epoch: int, model_manager: ModelManager) -> None:
        raise NotImplementedError()

    def epoch_end(self, epoch: int, model_manager: ModelManager) -> None:
        raise NotImplementedError()

    def get_state(self) -> Any:
        raise NotImplementedError()

    def set_state(self, state: Any) -> None:
        raise NotImplementedError()
