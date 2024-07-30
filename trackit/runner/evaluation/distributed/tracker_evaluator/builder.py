from trackit.core.runtime.build_context import BuildContext


def build_tracker_evaluator(evaluator_config: dict, config: dict, build_context: BuildContext):
    if evaluator_config['type'] == 'default':
        from .default.builder import build_default_tracker_evaluator
        return build_default_tracker_evaluator(evaluator_config, config, build_context)
    else:
        raise NotImplementedError("Unknown tracker evaluator type: {}".format(evaluator_config['type']))
