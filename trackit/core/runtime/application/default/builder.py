import os.path
import torch
import numpy as np
from typing import Optional, Dict, Sequence
from types import MappingProxyType
import logging
from .local_context import (ApplicationTaskDescription, ApplicationDataContext, ApplicationRunnerContext,
                            ApplicationContext, GlobalIterationCounter, EpochIterator)
from .global_context_manager import GlobalContextManager
from . import DefaultApplication
from trackit.models.methods.builder import get_model_build_context
from trackit.models import ModelManager
from trackit.core.runtime.build_context import BuildContext
from trackit.core.runtime.context.task import TaskContext
from trackit.core.runtime.utils.checkpoint.save.builder import build_checkpoint_dumper
from trackit.core.runtime.utils.epoch_activation_criteria.builder import build_epoch_activation_criteria
from trackit.core.runtime.metric_logger.builder import build_metric_logger
from trackit.core.runtime.metric_logger.adaptors.local import LocalMetricLoggerWrapper
from trackit.data.builder import build_data_pipeline
from trackit.data.context import DataContext
from trackit.data import HostDataPipeline
from trackit.runner.builder import build_runner
from trackit.runner.context import RunnerContext
from trackit.miscellanies.torch.distributed import is_dist_initialized, is_main_process
from .model_efficiency_assessment import ModelEfficiencyAssessment


logger = logging.getLogger(__name__)


def _get_build_context(name: str, model_manager: ModelManager, runtime_vars,
                       global_synchronized_rng: np.random.Generator,
                       local_rng: np.random.Generator,
                       instance_specific_rng: np.random.Generator,
                       wandb_instance):
    seed = runtime_vars.seed
    pin_memory = runtime_vars.pin_memory
    master_node_ip_addr = runtime_vars.master_address
    return BuildContext(name, model_manager, torch.device(runtime_vars.device),
                        seed, pin_memory, master_node_ip_addr, runtime_vars.run_id,
                        global_synchronized_rng, local_rng, instance_specific_rng, wandb_instance)


def _build_batch_collective_communication(branch_config: dict):
    from trackit.core.runtime.services.batch_collective_communication import BatchCollectiveCommunication_FixedTimeInterval, TimedBatchCollectiveCommunication_FixedStepInterval
    batch_collective_communication_config = branch_config.get('batch_collective_communication', None)
    if batch_collective_communication_config is None or not is_dist_initialized():
        return TimedBatchCollectiveCommunication_FixedStepInterval(step_interval=1)
    else:
        batch_collective_communication_type = batch_collective_communication_config['type']
        if batch_collective_communication_type == 'time':
            return BatchCollectiveCommunication_FixedTimeInterval(
                batch_collective_communication_config['interval'])
        elif batch_collective_communication_type == 'step':
            return TimedBatchCollectiveCommunication_FixedStepInterval(
                batch_collective_communication_config['interval'])
        else:
            raise NotImplementedError(f'Unknown batch collective communication type: {batch_collective_communication_type}')


def _build_task(task_name: str, branch_config: dict, output_path: Optional[str], num_epochs: int,
                context_manager: GlobalContextManager, build_context: BuildContext):
    if output_path is not None:
        output_path = os.path.join(output_path, task_name)
    epoch_activation_criteria = build_epoch_activation_criteria(branch_config.get('epoch_trigger', None), num_epochs)
    is_train = branch_config['is_train']
    data_name = branch_config['data']
    runner_name = branch_config['runner']

    metric_logger = build_metric_logger(branch_config.get('logging', None), build_context)
    local_metric_logger = metric_logger.get_logger('local')
    assert isinstance(local_metric_logger, LocalMetricLoggerWrapper)

    batch_collective_communication = _build_batch_collective_communication(branch_config)

    task_desc = ApplicationTaskDescription(task_name, data_name, runner_name, epoch_activation_criteria,
                                           metric_logger, local_metric_logger, is_train, build_context.services,
                                           batch_collective_communication)

    context = TaskContext(task_name, is_train, output_path, data_name, runner_name, epoch_activation_criteria,
                          MappingProxyType(build_context.variables), batch_collective_communication)
    context_manager.set_task_context(task_name, context)

    return task_desc


def _build_model_efficiency_assessor(config: dict, runtime_vars, wandb_instance):
    model_efficiency_assessor = None
    model_assessment_config = config['run']['efficiency_assessment']
    if model_assessment_config.get('enabled', False):
        model_assessment_kwargs = {}
        if 'latency' in model_assessment_config and 'auto_mixed_precision' in model_assessment_config['latency']:
            model_assessment_kwargs['latency_test_enable_amp'] = model_assessment_config['latency']['auto_mixed_precision'].get('enabled', False)
            dtype = model_assessment_config['latency']['auto_mixed_precision'].get('dtype', 'float16')
            if dtype == 'float16':
                dtype = torch.float16
            elif dtype == 'bfloat16':
                dtype = torch.bfloat16
            else:
                raise NotImplementedError(f'Unknown dtype: {dtype}')
            model_assessment_kwargs['latency_test_amp_dtype'] = dtype
        model_efficiency_assessor = ModelEfficiencyAssessment(torch.device(runtime_vars.device), wandb_instance,
                                                              **model_assessment_kwargs)
    return model_efficiency_assessor


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
    model_manager = ModelManager(get_model_build_context(config))

    model_efficiency_assessor = _build_model_efficiency_assessor(config, runtime_vars, wandb_instance)

    num_epochs = config['run']['num_epochs']

    # build checkpoint
    checkpoint_dumper = None
    if runtime_vars.output_dir is not None:
        checkpoint_output_path = os.path.join(runtime_vars.output_dir, 'checkpoint')
        if is_main_process():
            os.makedirs(checkpoint_output_path)
        checkpoint_dumper = build_checkpoint_dumper(config['run']['checkpoint'], checkpoint_output_path, num_epochs)

    # build task description
    context_manager = GlobalContextManager()
    tasks: Dict[str, ApplicationTaskDescription] = {}
    for task_name, task_config in config['run']['task'].items():
        logger.info(f'start build task: {task_name}')
        build_context = _get_build_context(name, model_manager, runtime_vars,
                                           global_synchronized_rng, local_rng, instance_specific_rng,
                                           wandb_instance)
        branch_task_desc = _build_task(task_name, task_config, runtime_vars.output_dir, num_epochs, context_manager,
                                       build_context)
        tasks[task_name] = branch_task_desc
        logger.info(f'finish build task: {task_name}')

    # build data
    data_inputs: Dict[str, ApplicationDataContext] = {}
    data_host_pipelines: Dict[str, Sequence[HostDataPipeline]] = {}
    for data_name, data_config in config['run']['data'].items():
        logger.info(f'start build data pipeline: {data_name}')
        build_context = _get_build_context(name, model_manager, runtime_vars,
                                           global_synchronized_rng, local_rng, instance_specific_rng,
                                           wandb_instance)
        data_pipeline = build_data_pipeline(data_name, data_config, build_context, config)
        context_manager.set_data_context(data_name, DataContext(data_name, MappingProxyType(build_context.variables)))
        batch_size = None
        if 'batch_size' in build_context.variables:
            batch_size = build_context.variables['batch_size']
        application_data_context = ApplicationDataContext(data_name, batch_size, build_context.services,
                                                          data_pipeline.input)
        data_inputs[data_name] = application_data_context
        if data_pipeline.host is not None:
            data_host_pipelines[data_name] = data_pipeline.host
        logger.info(f'finish build data pipeline: {data_name}')

    # build runner
    runners: Dict[str, ApplicationRunnerContext] = {}
    for runner_name, runner_config in config['run']['runner'].items():
        logger.info(f'start build runner: {runner_name}')
        associated_task_contexts = {}
        associated_data_contexts = {}

        for task_name, task_context in context_manager.get_task_context_iterator():
            if task_context.runner_name == runner_name:
                associated_task_contexts[task_name] = task_context
                data_name = task_context.data_name
                if data_name not in associated_data_contexts:
                    associated_data_contexts[data_name] = context_manager.get_data_context(data_name)

        assert len(associated_task_contexts) > 0, f'runner {runner_name} is not associated with any task'
        build_context = _get_build_context(name, model_manager, runtime_vars,
                                           global_synchronized_rng, local_rng, instance_specific_rng,
                                           wandb_instance)
        runner = build_runner(runner_name, runner_config, build_context, associated_task_contexts,
                              associated_data_contexts, config)
        for task_name, task_context in associated_task_contexts.items():
            data_name = task_context.data_name
            if data_name in data_host_pipelines:
                this_data_host_pipelines = data_host_pipelines[data_name]
                for data_host_pipeline in this_data_host_pipelines:
                    runner.register_data_pipeline(data_name, data_host_pipeline)

        context_manager.set_runner_context(runner_name, RunnerContext(runner_name,
                                                                      MappingProxyType(build_context.variables)))

        runners[runner_name] = ApplicationRunnerContext(runner_name, build_context.services, runner)
        logger.info(f'finish build runner: {runner_name}')

    epoch_counter = EpochIterator(num_epochs)
    iteration_counter = GlobalIterationCounter()
    application_context = ApplicationContext(data_inputs, runners, tasks, epoch_counter, iteration_counter)

    logger.info('Application context is built, ready to run')

    return DefaultApplication(name, model_manager, model_efficiency_assessor, context_manager, application_context,
                              checkpoint_dumper, runtime_vars.weight_path, runtime_vars.resume)
