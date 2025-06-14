from typing import Mapping, Optional, Tuple

from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.context.task import TaskContext
from trackit.data.context import DataContext
from trackit.criteria.builder import build_criterion

from ..common.distributed_data_parallel.builder import get_distributed_data_parallel_option
from ..common.optimization.builder import build_default_optimization_modules
from ..common.torch_compile import TorchCompileOptions
from .runner import DefaultTrainer


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


def build_default_training_runner(runner_config: dict, build_context: BuildContext, config: dict,
                                  associated_task_contexts: Mapping[str, TaskContext],
                                  associated_data_contexts: Mapping[str, DataContext]):
    train_task_context, train_data_context = _get_train_task_data_context(associated_task_contexts, associated_data_contexts)
    assert train_task_context is not None, "a train task attached to this runner is required"

    torch_compile_options = TorchCompileOptions.from_config(runner_config.get('torch_compile', None))

    device = build_context.device
    dtype = train_data_context.dtype
    model = build_context.model.create(device, dtype)

    num_epochs = train_task_context.epoch_selector.total_executions()
    train_batch_size = train_data_context.variables['batch_size']
    num_iterations_per_epoch = train_data_context.variables['num_iterations_per_epoch']

    criterion = build_criterion(runner_config['criteria'], device, dtype)

    optimization_modules_and_options = build_default_optimization_modules(
        model.model, criterion, runner_config, train_batch_size, num_epochs, num_iterations_per_epoch,
        build_context.device, torch_compile_options)

    ddp_option = get_distributed_data_parallel_option(runner_config, model.model, criterion, build_context.device,
                                                      optimization_modules_and_options.grad_accumulation_steps)

    detect_unused_parameters = runner_config['detect_unused_parameters']
    print_per_parameter_statistics = runner_config.get('print_per_parameter_statistics', False)
    train_mode_during_validation = runner_config.get('train_mode_during_validation', False)

    return DefaultTrainer(model, criterion, optimization_modules_and_options,
                          ddp_option, torch_compile_options, detect_unused_parameters, print_per_parameter_statistics,
                          train_mode_during_validation)
