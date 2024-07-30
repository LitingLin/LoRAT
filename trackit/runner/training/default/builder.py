import torch
import sys
from typing import Mapping, Optional, Tuple

from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.context.task import TaskContext
from trackit.data.context import DataContext
from trackit.criteria.builder import build_criterion

from trackit.miscellanies.operating_system import get_os_running_on, OperatingSystem

from trackit.runner.training.common.distributed_data_parallel.builder import get_distributed_data_parallel_option
from trackit.runner.training.default import DefaultTrainer
from trackit.runner.training.common.optimization.builder import build_default_optimization_modules


def _get_train_task_data_context(associated_task_contexts: Mapping[str, TaskContext],
                                 associated_data_contexts: Mapping[str, DataContext]) -> Tuple[Optional[TaskContext], Optional[DataContext]]:
    train_task_context = None
    train_data_context = None
    for task_context in associated_task_contexts.values():
        if task_context.is_train:
            assert train_task_context is None, "limitation: only support one train branch"
            train_task_context = task_context
            train_data_context = associated_data_contexts[task_context.data_name]
    return train_task_context, train_data_context


def build_torch_compile_options(config: dict) -> Optional[dict]:
    torch_compile_config = config['torch_compile']
    enabled = torch_compile_config['enabled']
    if int(torch.__version__.split('.')[0]) == 1:
        print("torch.compile is not supported for PyTorch 1.x", file=sys.stderr)
        enabled = False
    if enabled and get_os_running_on() != OperatingSystem.Linux:  # workaround: to remove when pytorch supports
        print('Only Linux is supported for torch.compile, disabled', file=sys.stderr)
        enabled = False

    if not enabled:
        return None
    if 'parameters' in torch_compile_config:
        return torch_compile_config['parameters']
    else:
        return {}


def build_default_training_runner(runner_config: dict, build_context: BuildContext,
                                  associated_task_contexts: Mapping[str, TaskContext],
                                  associated_data_contexts: Mapping[str, DataContext]):
    train_task_context, train_data_context = _get_train_task_data_context(associated_task_contexts, associated_data_contexts)
    assert train_task_context is not None, "limitation: a train task attached to this runner is required"

    torch_compile_options = build_torch_compile_options(runner_config)

    model = build_context.model.create(build_context.device)

    num_epochs = len(train_task_context.epoch_activation_criteria)
    train_batch_size = train_data_context.variables['batch_size']
    num_iterations_per_epoch = train_data_context.variables['num_iterations_per_epoch']

    num_total_iterations = num_epochs * num_iterations_per_epoch

    criterion = build_criterion(runner_config['criteria'], build_context, num_total_iterations)
    criterion.to(build_context.device)

    optimization_modules_and_options = build_default_optimization_modules(
        model.model, criterion, runner_config, train_batch_size, num_epochs, num_iterations_per_epoch,
        build_context.device)

    ddp_option = get_distributed_data_parallel_option(runner_config, model.model, criterion, build_context.device,
                                                      torch_compile_options is not None)

    detect_unused_parameters = runner_config['detect_unused_parameters']

    return DefaultTrainer(model, criterion, optimization_modules_and_options,
                          ddp_option, torch_compile_options, detect_unused_parameters)
