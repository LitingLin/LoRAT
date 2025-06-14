from trackit.data.protocol.eval_input import TrackerEvalData
from ..types import TrackerEvaluationPipeline_Context, TrackingPipeline_ResultHolder
from ... import EvaluatorContext


class TrackingPipeline:
    def start(self, evaluator_context: EvaluatorContext, global_objects: dict):
        pass

    def stop(self, evaluator_context: EvaluatorContext, global_objects: dict):
        pass

    def initialize(self, data: TrackerEvalData, model, context: TrackerEvaluationPipeline_Context):
        raise NotImplementedError

    def track(self, data: TrackerEvalData, model, context: TrackerEvaluationPipeline_Context,
              result: TrackingPipeline_ResultHolder):
        raise NotImplementedError
