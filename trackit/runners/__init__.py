from typing import Any, Optional

from trackit.data.context import DataContext
from trackit.models import ModelManager
from trackit.data import MainProcessDataPipeline


class Runner:
    def register_data_pipeline(self, task_name: Optional[str], data_pipeline: MainProcessDataPipeline) -> None:
        raise NotImplementedError()

    def run(self, data) -> None:
        raise NotImplementedError()

    def epoch_begin(self, epoch: int, task_name: Optional[str], is_train: bool,
                    model_manager: ModelManager, data_context: DataContext) -> None:
        raise NotImplementedError()

    def epoch_end(self, epoch: int, task_name: Optional[str], is_train: bool,
                  model_manager: ModelManager, data_context: DataContext) -> None:
        raise NotImplementedError()

    def get_state(self, checkpoint_path: str) -> Any:
        return None

    def set_state(self, state: Any, checkpoint_path: str) -> None:
        pass
