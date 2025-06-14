from trackit.core.runtime.build_context import BuildContext
from . import DefaultTrackerEvaluator
from .pipelines.builder import build_tracker_evaluator_pipeline


def build_default_tracker_evaluator(evaluator_config: dict, config: dict, build_context: BuildContext):
    evaluator = DefaultTrackerEvaluator(
        build_tracker_evaluator_pipeline(evaluator_config['pipeline'], config, build_context.device,
                                         build_context.num_epochs))

    return evaluator
