import gc
import sys
import logging
from typing import Optional, Sequence

from tqdm import tqdm

from trackit.miscellanies.torch.distributed import get_world_size
from trackit.miscellanies.torch.distributed.barrier import torch_distributed_barrier
from trackit.models import ModelManager
from trackit.core.runtime.services.event import emit_iteration_begin_event, emit_iteration_end_event, emit_epoch_begin_event, emit_epoch_end_event, emit_start_event, emit_stop_event
from trackit.core.runtime.metric_logger import enable_metric_logger
from trackit.core.runtime.utils.checkpoint.save import CheckpointDumper
from trackit.core.runtime.utils.checkpoint.load import load_application_state
from trackit.core.runtime.metric_logger.epoch_metric import enable_epoch_metrics, disable_epoch_metrics, EpochMetrics
from .early_stopping import EarlyStoppingManager
from .global_context_manager import GlobalContextManager
from .local_context import ApplicationTaskContext, ApplicationDataContext, ApplicationRunnerContext, GlobalIterationCounter, EpochIterator, ApplicationContext
from .checkpoint_dumper import ApplicationCheckpointDumper

logger = logging.getLogger(__name__)


def run_task(model_manager: ModelManager,
             task_context: ApplicationTaskContext,
             data_context: ApplicationDataContext,
             runner_context: ApplicationRunnerContext,
             epoch: int, global_step_counter: GlobalIterationCounter,
             checkpoint_dumper: ApplicationCheckpointDumper):
    all_event_registries = (task_context.services_registry.event,
                            runner_context.services_registry.event,
                            data_context.services_registry.event)
    all_batch_collective_communication_registries = (task_context.services_registry.batch_collective_communication,
                                                     runner_context.services_registry.batch_collective_communication,
                                                     data_context.services_registry.batch_collective_communication)

    task_name = task_context.context.name
    is_train = task_context.context.is_train
    runner = runner_context.runner

    local_metric_logger = task_context.local_metric_logger
    metric_logger = task_context.metric_logger
    collective_communication = task_context.context.collective_communication
    garbage_collection = task_context.garbage_collection

    metric_logger.set_step(global_step_counter.get_sample_processed())
    with enable_metric_logger(metric_logger, local_metric_logger):
        runner.epoch_begin(epoch, task_name, is_train, model_manager, data_context.context)
        emit_epoch_begin_event(all_event_registries, epoch, is_train)

        logger.info('waiting for all processes to be ready: begin barrier')
        torch_distributed_barrier()
        logger.info('waiting for all processes to be ready: end barrier')

        collective_communication.begin(all_batch_collective_communication_registries)
        garbage_collection.begin()

        for data in checkpoint_dumper.dump_every_iteration(
                local_metric_logger.log_every(data_context.data_input_pipeline)):
            metric_logger.set_step(global_step_counter.get_sample_processed())
            emit_iteration_begin_event(all_event_registries, is_train)
            runner.run(data)
            emit_iteration_end_event(reversed(all_event_registries), is_train)

            collective_communication.run()

            metric_logger.commit()
            if is_train:
                global_step_counter.update(_get_batch_size(data, data_context))
            garbage_collection.run()

        garbage_collection.end()
        collective_communication.end()  # may call run() as well

        logger.info('waiting for all processes to be ready: begin barrier')
        torch_distributed_barrier()
        logger.info('waiting for all processes to be ready: end barrier')

        emit_epoch_end_event(reversed(all_event_registries), epoch, is_train)
        runner.epoch_end(epoch, task_name, is_train, model_manager, data_context.context)


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
                 context_manager: GlobalContextManager,
                 all_contexts: ApplicationContext,
                 checkpoint_dumper: Optional[CheckpointDumper],
                 early_stopping_manager: Optional[EarlyStoppingManager],
                 model_weight_file: Optional[Sequence[str]] = None,
                 application_state_file: Optional[str] = None):
        self._model_name = model_name
        self._model_manager = model_manager
        self._context_manager = context_manager
        self._all_context = all_contexts
        self._checkpoint_dumper = checkpoint_dumper
        self._early_stopping_manager = early_stopping_manager
        self._model_weight_file = model_weight_file
        self._application_state_file = application_state_file

    def run(self):
        if self._model_weight_file is not None:
            for model_weight_file in self._model_weight_file:
                self._model_manager.load_state_dict_from_file(model_weight_file)
                print(f'loaded model weight from {model_weight_file}', flush=True)
        if self._application_state_file is not None:
            if self._model_weight_file is None:
                print('warn: state resumed but model weight file is not specified', flush=True)
            load_application_state(self._all_context.load_state_dict, self._application_state_file)
            print(f'loaded state from {self._application_state_file}', flush=True)

        epoch_metrics_holder = EpochMetrics()
        checkpoint_dumper = ApplicationCheckpointDumper(self._checkpoint_dumper,
                                                        self._all_context.iteration,
                                                        self._model_manager,
                                                        self._all_context.state_dict,
                                                        epoch_metrics_holder)
        epoch_iterator = self._all_context.epoch
        global_step_counter = self._all_context.iteration
        enable_epoch_metrics(epoch_metrics_holder)
        emit_start_event(_get_all_event_listener_registries(self._all_context))
        has_train_task = any(task.context.is_train for task in self._all_context.tasks.values())
        early_stopped = False
        try:
            for epoch in checkpoint_dumper.dump_every_epoch(
                    tqdm(epoch_iterator, desc=f'Train {self._model_name}' if has_train_task else f'Eval {self._model_name}',
                         file=sys.stdout, position=0, leave=True,
                         initial=epoch_iterator.get_current())):
                print()
                for task_name, task_context in self._all_context.tasks.items():
                    data_context = self._all_context.data_inputs[task_context.context.data_name]
                    runner_context = self._all_context.runners[task_context.context.runner_name]

                    if task_context.context.epoch_selector.should_execute(epoch):
                        self._context_manager.activate(task_name, epoch)
                        run_task(self._model_manager, task_context, data_context, runner_context, epoch, global_step_counter, checkpoint_dumper)
                        self._context_manager.finalize()
                        checkpoint_dumper.dump_model_state()
                        # Explicitly trigger garbage collection after completing a task.
                        # REASON: To prevent a common and hard-to-debug CUDA error. If this task left
                        # uncollected CUDA tensors, and the next task's DataLoader forks new worker
                        # processes, those workers could try to garbage-collect the old tensors.
                        # This fails because forked processes do not properly inherit the CUDA context,
                        # leading to a crash. This line ensures all cleanup happens here, in the main process
                        gc.collect()
                        if self._early_stopping_manager is not None and not early_stopped:
                            early_stopped = self._early_stopping_manager.should_stop(epoch_metrics_holder.get(epoch), epoch)
                            if early_stopped:
                                epoch_iterator.set_total_epochs(epoch + 1)
        finally:
            emit_stop_event(reversed(_get_all_event_listener_registries(self._all_context)))
            disable_epoch_metrics()


def _get_batch_size(data, data_context: ApplicationDataContext) -> int:
    from trackit.data.protocol.train_input import TrainData
    if isinstance(data, TrainData):
        return data.batch_size * get_world_size() # assume that batch size is the same for all processes
    if data_context.batch_size is not None:
        return data_context.batch_size * get_world_size()
    return 1
