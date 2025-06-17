import gc
from typing import Optional

import torch
import torch._C

from trackit.miscellanies.torch.check_version import is_torch_version_greater_or_equal
from trackit.runners import Runner
from trackit.models import ModelManager
from trackit.models.compiling import InferenceEngine, OptimizedModel
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.data.context import get_current_data_context
from trackit.data import MainProcessDataPipeline
from .tracker_evaluator import TrackerEvaluator, EvaluatorContext


class DefaultTrackerEvaluationRunner(Runner):
    def __init__(self, tracker_evaluator: TrackerEvaluator,
                 inference_engine: InferenceEngine, device: torch.device):
        self.tracker_evaluator = tracker_evaluator
        self.inference_engine = inference_engine
        self._device = device

        self.data_pipelines_on_main_process = {}
        self.task_name = None

        self.optimized_model: Optional[OptimizedModel] = None

    def epoch_begin(self, epoch: int, task_name: str, is_train: bool,
                    model_manager: ModelManager, data_context):
        self.task_name = task_name
        assert not is_train, "Evaluator can only be run in evaluation mode"
        max_batch_size = get_current_data_context().variables['batch_size']
        num_workers = get_current_data_context().variables['num_workers']

        self.optimized_model = self.inference_engine(model_manager, self._device, data_context.dtype, max_batch_size, num_workers)
        self.evaluator_context = EvaluatorContext(
            epoch=epoch,
            max_batch_size=max_batch_size,
            num_input_data_streams=num_workers,
            dtype=data_context.dtype,
            auto_mixed_precision_dtype=self.optimized_model.auto_mixed_precision_dtype,
            model=self.optimized_model.raw_model,
        )
        self.tracker_evaluator.start(self.evaluator_context)
        data_pipeline_on_main_process = self.data_pipelines_on_main_process.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data_pipeline.start(epoch, self.optimized_model.raw_model)

    def epoch_end(self, epoch: int, *_):
        data_pipeline_on_main_process = self.data_pipelines_on_main_process.get(self.task_name, None)
        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data_pipeline.stop(epoch)
        self.tracker_evaluator.stop(self.evaluator_context)
        self.optimized_model = None
        del self.evaluator_context
        gc.collect()
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
            if is_torch_version_greater_or_equal((2, 5)):
                torch._C._host_emptyCache()

    def run(self, data: TrackerEvalData):
        assert self.task_name is not None
        data_pipeline_on_main_process = self.data_pipelines_on_main_process.get(self.task_name, None)

        if data_pipeline_on_main_process is not None:
            for data_pipeline in data_pipeline_on_main_process:
                data = data_pipeline.pre_process(data)

        outputs = self.tracker_evaluator.run(data, self.optimized_model.model, self.optimized_model.raw_model)

        if data_pipeline_on_main_process is not None:
            for data_pipeline in reversed(data_pipeline_on_main_process):
                outputs = data_pipeline.post_process(outputs)

    def register_data_pipeline(self, task_name: str, data_pipeline: MainProcessDataPipeline) -> None:
        if task_name not in self.data_pipelines_on_main_process:
            self.data_pipelines_on_main_process[task_name] = []
        self.data_pipelines_on_main_process[task_name].append(data_pipeline)
