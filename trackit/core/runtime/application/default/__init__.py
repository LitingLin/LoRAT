from typing import Optional, Sequence
from tqdm import tqdm
import sys
import logging

from trackit.miscellanies.torch.distributed import get_world_size
from trackit.miscellanies.torch.distributed.barrier import torch_distributed_barrier
from trackit.models import ModelManager
from trackit.core.runtime.services.event import emit_iteration_begin_event, emit_iteration_end_event, emit_epoch_begin_event, emit_epoch_end_event, emit_start_event, emit_stop_event
from trackit.core.runtime.metric_logger import enable_metric_logger
from trackit.core.runtime.utils.checkpoint.save import CheckpointDumper
from trackit.core.runtime.utils.checkpoint.load import load_application_state
from trackit.core.runtime.metric_logger.epoch_metric import get_current_epoch_metrics, enable_epoch_metrics, disable_epoch_metrics, EpochMetrics
from .global_context_manager import GlobalContextManager
from .local_context import ApplicationTaskDescription, ApplicationDataContext, ApplicationRunnerContext, GlobalIterationCounter, EpochIterator, ApplicationContext
from .model_efficiency_assessment import ModelEfficiencyAssessment

logger = logging.getLogger(__name__)


def run_task(model_manager: ModelManager, task_desc: ApplicationTaskDescription, data_context: ApplicationDataContext,
             runner_context: ApplicationRunnerContext,
             epoch: int, global_step_counter: GlobalIterationCounter):
    all_event_registries = (task_desc.services_registry.event,
                            runner_context.services_registry.event,
                            data_context.services_registry.event)
    all_batch_collective_communication_registries = (task_desc.services_registry.batch_collective_communication,
                                                     runner_context.services_registry.batch_collective_communication,
                                                     data_context.services_registry.batch_collective_communication)

    task_name = task_desc.name
    is_train = task_desc.is_train
    runner = runner_context.runner
    if data_context.batch_size is not None:
        batch_size = data_context.batch_size * get_world_size()
    else:
        batch_size = 1

    local_metric_logger = task_desc.local_metric_logger
    metric_logger = task_desc.metric_logger
    collective_communication = task_desc.collective_communication

    metric_logger.set_step(global_step_counter.get_iteration())
    with enable_metric_logger(metric_logger, local_metric_logger):
        runner.switch_task(task_name, is_train)
        runner.epoch_begin(epoch, model_manager)
        emit_epoch_begin_event(all_event_registries, epoch, is_train)

        logger.info('waiting for all processes to be ready: begin barrier')
        torch_distributed_barrier()
        logger.info('waiting for all processes to be ready: end barrier')

        collective_communication.begin(all_batch_collective_communication_registries)

        for data in local_metric_logger.log_every(data_context.data_input_pipeline):
            metric_logger.set_step(global_step_counter.get_iteration())
            emit_iteration_begin_event(all_event_registries, is_train)
            runner.run(data)
            emit_iteration_end_event(reversed(all_event_registries), is_train)

            collective_communication.run()

            metric_logger.commit()
            if is_train:
                global_step_counter.update(batch_size)

        collective_communication.end()  # may call run() as well

        logger.info('waiting for all processes to be ready: begin barrier')
        torch_distributed_barrier()
        logger.info('waiting for all processes to be ready: end barrier')

        emit_epoch_end_event(reversed(all_event_registries), epoch, is_train)
        runner.epoch_end(epoch, model_manager)


def _get_all_event_listener_registries(all_contexts: ApplicationContext):
    all_registries = []
    for branch in all_contexts.tasks.values():
        all_registries.append(branch.services_registry.event)
    for data_context in all_contexts.data_inputs.values():
        all_registries.append(data_context.services_registry.event)
    for runner_context in all_contexts.runners.values():
        all_registries.append(runner_context.services_registry.event)
    return all_registries


class DefaultApplication:
    def __init__(self, model_name: str,
                 model_manager: ModelManager,
                 model_efficiency_assessment: Optional[ModelEfficiencyAssessment],
                 context_manager: GlobalContextManager,
                 all_contexts: ApplicationContext,
                 checkpoint_dumper: Optional[CheckpointDumper],
                 model_weight_file: Optional[Sequence[str]] = None,
                 application_state_file: Optional[str] = None):
        self._model_name = model_name
        self._model_manager = model_manager
        self._model_efficiency_assessment = model_efficiency_assessment
        self._context_manager = context_manager
        self._all_context = all_contexts
        self._checkpoint_dumper = checkpoint_dumper
        self._model_weight_file = model_weight_file
        self._application_state_file = application_state_file

    def run(self):
        if self._model_efficiency_assessment is not None:
            self._model_efficiency_assessment(self._model_manager)
        if self._model_weight_file is not None:
            for model_weight_file in self._model_weight_file:
                self._model_manager.load_state_dict_from_file(model_weight_file, use_safetensors=True)
                print(f'loaded model weight from {model_weight_file}', flush=True)
        if self._application_state_file is not None:
            if self._model_weight_file is None:
                print('warn: state resumed but model weight file is not specified', flush=True)
            load_application_state(self._all_context.load_state_dict, self._application_state_file)
            print(f'loaded state from {self._application_state_file}', flush=True)

        enable_epoch_metrics(EpochMetrics())
        emit_start_event(_get_all_event_listener_registries(self._all_context))
        epoch_iterator = self._all_context.epoch
        has_train_task = any(task.is_train for task in self._all_context.tasks.values())
        for epoch in tqdm(epoch_iterator, desc=f'Train {self._model_name}' if has_train_task else f'Eval {self._model_name}',
                          file=sys.stdout, position=0, leave=True,
                          initial=epoch_iterator.get_current()):
            print()
            for task_name, task in self._all_context.tasks.items():
                data_context = self._all_context.data_inputs[task.data_name]
                runner_context = self._all_context.runners[task.runner_name]

                if task.epoch_activation_criteria(epoch):
                    self._context_manager.activate(task_name, epoch)
                    run_task(self._model_manager, task, data_context, runner_context, epoch, self._all_context.iteration)
                    self._context_manager.finalize()
                    if task.is_train:
                        if self._checkpoint_dumper is not None:
                            self._checkpoint_dumper.temporary_dump(epoch, self._model_manager.version, self._model_manager.state_dict)

            if self._checkpoint_dumper is not None:
                self._checkpoint_dumper.dump(epoch, get_current_epoch_metrics().get(epoch), self._model_manager.version,
                                             None, self._all_context.state_dict)
            else:
                print('Output path is not set. Skip checkpoint saving.')

        emit_stop_event(_get_all_event_listener_registries(self._all_context))
        disable_epoch_metrics()
