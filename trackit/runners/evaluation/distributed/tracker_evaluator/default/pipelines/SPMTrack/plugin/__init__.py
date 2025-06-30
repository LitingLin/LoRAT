from typing import Any
from trackit.data.protocol.eval_input import TrackerEvalData
from ....types import TrackerEvaluationPipeline_Context, TrackingPipeline_ResultHolder
from ..... import EvaluatorContext


class TrackingPipelinePlugin:
    def start(self, context: EvaluatorContext, global_objects: dict):
        pass

    def stop(self, context: EvaluatorContext, global_objects: dict):
        pass

    def prepare_initialization(self, data: TrackerEvalData, model_input_params: dict,
                               context: TrackerEvaluationPipeline_Context):
        pass

    def on_initialized(self, data: TrackerEvalData, model_outputs, context: TrackerEvaluationPipeline_Context):
        pass

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict,
                         context: TrackerEvaluationPipeline_Context):
        pass

    def on_tracked(self, data: TrackerEvalData, model_outputs: Any, context: TrackerEvaluationPipeline_Context,
                   result: TrackingPipeline_ResultHolder):
        pass
