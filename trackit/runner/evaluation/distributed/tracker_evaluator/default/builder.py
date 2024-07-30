from trackit.core.runtime.build_context import BuildContext
from . import DefaultTrackerEvaluator
from .pipelines.builder import build_tracker_evaluator_data_pipeline


def build_default_tracker_evaluator(evaluator_config: dict, config: dict, build_context: BuildContext):
    evaluator = DefaultTrackerEvaluator(
        build_tracker_evaluator_data_pipeline(evaluator_config, config, build_context.device))

    return evaluator
