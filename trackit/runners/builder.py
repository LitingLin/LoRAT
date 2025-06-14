from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.context.task import TaskContext
from trackit.data.context import DataContext
from trackit.miscellanies.printing import print_centered_text
from typing import Mapping


def build_runner(runner_name: str,
                 runner_config: dict, build_context: BuildContext,
                 associated_task_contexts: Mapping[str, TaskContext],
                 associated_data_contexts: Mapping[str, DataContext],
                 config: dict):
    print_centered_text(f"Building runner: {runner_name}")
    if runner_config['type'] == 'default_train':
        from .training.default.builder import build_default_training_runner
        runner = build_default_training_runner(runner_config, build_context, config, associated_task_contexts, associated_data_contexts)
    elif runner_config['type'] == 'default_eval':
        from .evaluation.distributed.builder import build_default_evaluation_runner
        runner = build_default_evaluation_runner(runner_config, build_context, config)
    elif runner_config['type'] == 'dummy':
        from .dummy import DummyRunner
        runner = DummyRunner()
    elif runner_config['type'] == 'deepspeed_train':
        from .training.with_deepspeed.builder import build_deepspeed_trainer
        runner = build_deepspeed_trainer(runner_config, build_context, config, associated_task_contexts, associated_data_contexts)
    else:
        raise NotImplementedError('Unknown runner type: {}'.format(runner_config['type']))
    print_centered_text('')
    return runner
