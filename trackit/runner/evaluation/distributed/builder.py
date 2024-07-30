from trackit.core.runtime.build_context import BuildContext
from trackit.models.compiling.builder import build_inference_engine
from . import DefaultTrackerEvaluationRunner
from .tracker_evaluator.builder import build_tracker_evaluator


def build_default_evaluation_runner(runner_config: dict, build_context: BuildContext, config: dict):
    tracker_evaluator = build_tracker_evaluator(runner_config['evaluator'], config, build_context)
    inference_engine = build_inference_engine(runner_config['inference_engine'])
    return DefaultTrackerEvaluationRunner(tracker_evaluator, inference_engine, build_context.device)
