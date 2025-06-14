from typing import Optional

from trackit.data import MainProcessDataPipeline
from trackit.runners import Runner


class DummyRunner(Runner):
    def __init__(self):
        self.data_pipelines = {}

    def register_data_pipeline(self, task_name: Optional[str], data_pipeline: MainProcessDataPipeline) -> None:
        if task_name not in self.data_pipelines:
            self.data_pipelines[task_name] = []
        self.data_pipelines[task_name].append(data_pipeline)

    def run(self, data) -> None:
        data_pipeline_on_main_process = self.data_pipelines.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data = data_pipeline.pre_process(data)
            model_output = None
            for data_pipeline in reversed(data_pipeline_on_main_process):
                model_output = data_pipeline.post_process(model_output)

    def epoch_begin(self, epoch: int, task_name: str, *_) -> None:
        self.task_name = task_name
        pass

    def epoch_end(self, *_) -> None:
        pass
