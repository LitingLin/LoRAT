import os.path
import torch
import numpy as np
from typing import Optional, Dict, Sequence
from types import MappingProxyType
import logging
from .local_context import (ApplicationTaskContext, ApplicationDataContext, ApplicationRunnerContext,
                            ApplicationContext, GlobalIterationCounter, EpochIterator)
from .global_context_manager import GlobalContextManager
from . import DefaultApplication
from trackit.models.methods.builder import create_model_build_context
from trackit.models import ModelManager
from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.context.task import TaskContext
from trackit.core.runtime.utils.checkpoint.save.builder import build_checkpoint_dumper
from trackit.core.runtime.utils.execution_trigger.builder import build_execution_trigger
from trackit.core.runtime.metric_logger.builder import build_metric_logger
from trackit.core.runtime.metric_logger.adaptors.local import LocalMetricLoggerWrapper
from trackit.data.builder import build_data_pipeline
from trackit.data.context import DataContext
from trackit.data import MainProcessDataPipeline
from trackit.runners.builder import build_runner
from trackit.runners.context import RunnerContext
from trackit.miscellanies.torch.distributed import is_dist_initialized, is_rank_0_process
from ..profiling.builder import build_model_profiler


logger = logging.getLogger(__name__)


def _get_build_context(name: str, model_manager: ModelManager, runtime_vars, num_epochs: int,
                       global_synchronized_rng: np.random.Generator,
                       local_rng: np.random.Generator,
                       instance_specific_rng: np.random.Generator,
                       wandb_instance):
    seed = runtime_vars.seed
    pin_memory = runtime_vars.pin_memory
    master_node_ip_addr = runtime_vars.master_address
    return BuildContext(name, model_manager, torch.device(runtime_vars.device),
                        seed, pin_memory, master_node_ip_addr, runtime_vars.run_id,
                        num_epochs,
                        global_synchronized_rng, local_rng, instance_specific_rng, wandb_instance)


def _build_batch_collective_communication(task_config: dict):
    from trackit.core.runtime.services.batch_collective_communication import BatchCollectiveCommunication_FixedTimeInterval, BatchCollectiveCommunication_FixedStepInterval
    batch_collective_communication_config = task_config.get('batch_collective_communication', None)
    if batch_collective_communication_config is None or not is_dist_initialized():
        return BatchCollectiveCommunication_FixedStepInterval(step_interval=1)
    else:
        batch_collective_communication_type = batch_collective_communication_config['type']
        if batch_collective_communication_type == 'time':
            return BatchCollectiveCommunication_FixedTimeInterval(
                batch_collective_communication_config['interval'])
        elif batch_collective_communication_type == 'step':
            return BatchCollectiveCommunication_FixedStepInterval(
                batch_collective_communication_config['interval'])
        else:
            raise NotImplementedError(f'Unknown batch collective communication type: {batch_collective_communication_type}')


def _build_garbage_collection(task_config: dict):
    from trackit.core.runtime.services.garbage_collection import GarbageCollection
    if 'garbage_collection' in task_config:
        garbage_collection_config = task_config['garbage_collection']
        return GarbageCollection(garbage_collection_config['type'], garbage_collection_config.get('interval', None))
    return GarbageCollection('auto')


def _build_task(task_name: str, task_config: dict, output_path: Optional[str], build_context: BuildContext):
    if output_path is not None:
        output_path = os.path.join(output_path, task_name)
    epoch_selector = build_execution_trigger(task_config.get('epoch_trigger', None), build_context.num_epochs)
    is_train = task_config['is_train']
    data_name = task_config['data']
    runner_name = task_config['runner']

    metric_logger = build_metric_logger(task_config.get('logging', None), build_context)
    local_metric_logger = metric_logger.get_logger('local')
    assert isinstance(local_metric_logger, LocalMetricLoggerWrapper)

    batch_collective_communication = _build_batch_collective_communication(task_config)
    garbage_collection = _build_garbage_collection(task_config)

    context = TaskContext(task_name, is_train,
                          output_path, data_name, runner_name, epoch_selector,
                          MappingProxyType(build_context.variables), batch_collective_communication)

    return context, metric_logger, local_metric_logger, garbage_collection


def build_default_application(config: dict, runtime_vars,
                              global_synchronized_rng: np.random.Generator,
                              local_rng: np.random.Generator,
                              instance_specific_rng: np.random.Generator,
                              wandb_instance):
    if 'name' in config:
        name = config['name']
    else:
        name = runtime_vars.method_name + '-' + runtime_vars.config_name

    # build model factory
    model_manager = ModelManager(create_model_build_context(config), runtime_vars.seed)

    model_profiler = build_model_profiler(config, runtime_vars, wandb_instance)
    if model_profiler is not None:
        model_profiler(model_manager)
        del model_profiler

    num_epochs = config['run']['num_epochs']

    # build checkpoint
    checkpoint_dumper = None
    if runtime_vars.output_dir is not None and 'checkpoint' in config['run']:
        checkpoint_output_path = os.path.join(runtime_vars.output_dir, 'checkpoint')
        if is_rank_0_process():
            os.makedirs(checkpoint_output_path)
        checkpoint_dumper = build_checkpoint_dumper(config['run']['checkpoint'], checkpoint_output_path, num_epochs)

    # build task description
    context_manager = GlobalContextManager()
    tasks: Dict[str, ApplicationTaskContext] = {}
    for task_name, task_config in config['run']['task'].items():
        logger.info(f'start build task: {task_name}')
        build_context = _get_build_context(name, model_manager, runtime_vars, num_epochs,
                                           global_synchronized_rng, local_rng, instance_specific_rng,
                                           wandb_instance)
        task_context, metric_logger, local_metric_logger, garbage_collection = (
            _build_task(task_name, task_config, runtime_vars.output_dir, build_context))
        context_manager.set_task_context(task_name, task_context)

        task_desc = ApplicationTaskContext(task_context,
                                           metric_logger, local_metric_logger,
                                           build_context.services,
                                           garbage_collection)

        tasks[task_name] = task_desc
        logger.info(f'finish build task: {task_name}')

    # build data
    data_inputs: Dict[str, ApplicationDataContext] = {}
    data_pipelines_on_main_process: Dict[str, Sequence[MainProcessDataPipeline]] = {}
    for data_pipeline_name, data_pipeline_config in config['run']['data'].items():
        logger.info(f'start build data pipeline: {data_pipeline_name}')
        build_context = _get_build_context(name, model_manager, runtime_vars, num_epochs,
                                           global_synchronized_rng, local_rng, instance_specific_rng,
                                           wandb_instance)
        data_pipeline = build_data_pipeline(data_pipeline_name, data_pipeline_config, build_context, config)
        dtype = build_context.variables['dtype']
        data_context = DataContext(data_pipeline_name,
                                   MappingProxyType(build_context.variables),
                                   dtype)
        context_manager.set_data_context(data_pipeline_name, data_context)
        batch_size = build_context.variables.get('batch_size', None)
        application_data_context = ApplicationDataContext(data_context,
                                                          batch_size,
                                                          build_context.services,
                                                          data_pipeline.input)
        data_inputs[data_pipeline_name] = application_data_context
        if data_pipeline.on_main_process is not None:
            data_pipelines_on_main_process[data_pipeline_name] = data_pipeline.on_main_process
        logger.info(f'finish build data pipeline: {data_pipeline_name}')

    # build runner
    runners: Dict[str, ApplicationRunnerContext] = {}
    for runner_name, runner_config in config['run']['runner'].items():
        logger.info(f'start build runner: {runner_name}')
        associated_task_contexts = {}
        associated_data_contexts = {}

        for task_name, task_context in context_manager.get_task_context_iterator():
            if task_context.runner_name == runner_name:
                associated_task_contexts[task_name] = task_context
                data_pipeline_name = task_context.data_name
                if data_pipeline_name not in associated_data_contexts:
                    associated_data_contexts[data_pipeline_name] = context_manager.get_data_context(data_pipeline_name)

        assert len(associated_task_contexts) > 0, f'runner {runner_name} is not associated with any task'
        build_context = _get_build_context(name, model_manager, runtime_vars, num_epochs,
                                           global_synchronized_rng, local_rng, instance_specific_rng,
                                           wandb_instance)
        runner = build_runner(runner_name, runner_config, build_context, associated_task_contexts,
                              associated_data_contexts, config)
        for task_name, task_context in associated_task_contexts.items():
            data_pipeline_name = task_context.data_name
            if data_pipeline_name in data_pipelines_on_main_process:
                this_data_pipelines_on_main_process = data_pipelines_on_main_process[data_pipeline_name]
                for data_pipeline_on_main_process in this_data_pipelines_on_main_process:
                    runner.register_data_pipeline(data_pipeline_name, data_pipeline_on_main_process)

        runner_context = RunnerContext(runner_name,
                                        MappingProxyType(build_context.variables))
        context_manager.set_runner_context(runner_name, runner_context)

        runners[runner_name] = ApplicationRunnerContext(runner_context, build_context.services, runner)
        logger.info(f'finish build runner: {runner_name}')

    epoch_counter = EpochIterator(num_epochs)
    iteration_counter = GlobalIterationCounter()
    application_context = ApplicationContext(data_inputs, runners, tasks, epoch_counter, iteration_counter)

    logger.info('Application context is built, ready to run')

    return DefaultApplication(name, model_manager, context_manager, application_context,
                              checkpoint_dumper, runtime_vars.weight_path, runtime_vars.resume)
