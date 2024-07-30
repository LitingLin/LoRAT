from typing import Optional, Any

import torch
from torch import nn
import gc

from trackit.runner import Runner
from trackit.models import ModelManager
from trackit.models.compiling import InferenceEngine
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.data.context import get_current_data_context
from trackit.data import HostDataPipeline
from .tracker_evaluator import TrackerEvaluator, run_tracker_evaluator


class DefaultTrackerEvaluationRunner(Runner):
    def __init__(self, tracker_evaluator: TrackerEvaluator,
                 inference_engine: InferenceEngine, device: torch.device):
        self.tracker_evaluator = tracker_evaluator
        self.inference_engine = inference_engine
        self._device = device

        self.data_pipeline_on_host = {}
        self.branch_name = None

        self.raw_model: Optional[nn.Module] = None
        self.optimized_model: Any = None

    def switch_task(self, task_name, is_train):
        self.branch_name = task_name
        assert not is_train, "Evaluator can only be run in evaluation mode"

    def epoch_begin(self, epoch: int, model_manager: ModelManager):
        max_batch_size = get_current_data_context().variables['batch_size']

        self.optimized_model, self.raw_model = self.inference_engine(model_manager, self._device, max_batch_size)
        self.tracker_evaluator.on_epoch_begin()

    def epoch_end(self, epoch: int, _):
        self.tracker_evaluator.on_epoch_end()
        self.optimized_model = self.raw_model = None
        gc.collect()

    def run(self, data: TrackerEvalData):
        assert self.branch_name is not None
        data_pipeline_on_host = self.data_pipeline_on_host.get(self.branch_name, None)

        if data_pipeline_on_host is not None:
            for data_pipeline in data_pipeline_on_host:
                data = data_pipeline.pre_process(data)

        outputs = run_tracker_evaluator(self.tracker_evaluator, data, self.optimized_model, self.raw_model)

        if data_pipeline_on_host is not None:
            for data_pipeline in reversed(data_pipeline_on_host):
                outputs = data_pipeline.post_process(outputs)

    def register_data_pipeline(self, task_name: str, data_pipeline: HostDataPipeline) -> None:
        if task_name not in self.data_pipeline_on_host:
            self.data_pipeline_on_host[task_name] = []
        self.data_pipeline_on_host[task_name].append(data_pipeline)
