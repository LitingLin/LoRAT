from typing import Any
import torch.nn as nn

from trackit.data.protocol.eval_input import TrackerEvalData
from ..types import TrackerEvaluationPipeline_Context, TrackingPipeline_ResultHolder
from ... import EvaluatorContext


class TrackingPipeline:
    def start(self, evaluator_context: EvaluatorContext, global_objects: dict):
        pass

    def stop(self, evaluator_context: EvaluatorContext, global_objects: dict):
        pass

    def initialize(self, data: TrackerEvalData, model: Any, context: TrackerEvaluationPipeline_Context,
                   raw_model: nn.Module):
        raise NotImplementedError

    def track(self, data: TrackerEvalData, model: Any, context: TrackerEvaluationPipeline_Context,
              result: TrackingPipeline_ResultHolder, raw_model: nn.Module):
        raise NotImplementedError
