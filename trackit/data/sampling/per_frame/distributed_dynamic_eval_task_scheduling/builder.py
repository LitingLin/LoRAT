from typing import Sequence, Optional
import numpy as np
from tabulate import tabulate
from trackit.data.source import TrackingDataset
from trackit.core.runtime.build_context import BuildContext
from .distributed_evaluation_task_scheduler import DistributedTrackerEvaluationTaskDynamicSchedulerServer, DistributedTrackerEvaluationTaskDynamicSchedulerClient, SequenceOrder, DistributedTrackerEvaluationTaskDynamicScheduler
from trackit.data.utils.data_source_matcher.builder import build_data_source_matcher
from trackit.miscellanies.torch.distributed import is_rank_0_process, get_world_size
from trackit.data.context.variable.eval import DatasetEvaluationTask


def _get_dataset_repeat_times(repeat_times_rule: Optional[Sequence[dict]], datasets: Sequence[TrackingDataset]):
    repeat_times = np.ones(len(datasets), dtype=np.int32)
    if repeat_times_rule is not None:
        matcher_value_pairs = []
        for repeat_time_rule in repeat_times_rule:
            matcher = build_data_source_matcher(repeat_time_rule['match'])
            repeat_value = repeat_time_rule['value']
            matcher_value_pairs.append((matcher, repeat_value))

        for index_of_dataset, dataset in enumerate(datasets):
            for matcher, repeat_value in matcher_value_pairs:
                if matcher(dataset.get_name(), dataset.get_data_split()):
                    repeat_times[index_of_dataset] = repeat_value
                    break

    return repeat_times


def _build_eval_task_context_variable(datasets: Sequence[TrackingDataset], repeat_times: np.ndarray):
    eval_tasks = []
    for dataset, this_dataset_repeat_times in zip(datasets, repeat_times):
        track_names = set()
        for sequence in dataset:
            for track in sequence.get_track_iterator():
                track_name = track.get_name()
                assert track_name not in track_names
                track_names.add(track_name)
        eval_tasks.append(DatasetEvaluationTask(dataset.get_name(), dataset.get_data_split(), dataset.get_full_name(), track_names, this_dataset_repeat_times.item()))
    return tuple(eval_tasks)


def _print_dynamic_scheduler_info(server_address: str, datasets, repeat_times):
    string = 'dynamic evaluation task scheduler:\n'
    string += f'Orchestrator server address: {server_address}\n'
    string += tabulate(((dataset.get_full_name(), len(dataset), repeat_time) for dataset, repeat_time in zip(datasets, repeat_times)), headers=('dataset', 'num_sequence', 'repeat_times'))
    print(string)


def _print_summary_stat(number_of_evaluation_tasks: int, number_of_evaluation_frames: int, max_batch_size: int):
    string = '\t'.join((f'total number of sequences: {number_of_evaluation_tasks}',
                        f'total number of frames: {number_of_evaluation_frames}',
                        f'max batch size: {max_batch_size}'))

    print(string)


def adjust_max_batch_size(number_of_tasks: int, max_batch_size: int, world_size: int, number_of_workers_per_rank: int,):
    if number_of_tasks // (world_size * number_of_workers_per_rank) < max_batch_size:
        max_batch_size = number_of_tasks // (world_size * number_of_workers_per_rank)
        if max_batch_size == 0:
            max_batch_size = 1
        print(f'Warning: max_batch_size is too large, set to {max_batch_size} to ensure each rank has at least one task.')

    return max_batch_size


def build_distributed_tracker_evaluation_task_dynamic_scheduler(datasets: Sequence[TrackingDataset], sampler_config: dict, max_batch_size: int, number_of_workers: int, build_context: BuildContext):
    assert sampler_config['type'] == 'distributed_dynamic_scheduling'
    persistent = True

    if 'listen_address' not in sampler_config:
        sampling_orchestrator_server_address = f'tcp://{build_context.master_node_ip_addr}:{build_context.instance_specific_rng.integers(10000, 50000)}'
    else:
        sampling_orchestrator_server_address = sampler_config['listen_address']
    key = build_context.run_id

    repeat_times = _get_dataset_repeat_times(sampler_config.get('repeat_times', None), datasets)
    _print_dynamic_scheduler_info(sampling_orchestrator_server_address, datasets, repeat_times)

    order = SequenceOrder.random if sampler_config['shuffle'] else SequenceOrder.sequential

    world_size = get_world_size()
    scheduler = DistributedTrackerEvaluationTaskDynamicScheduler(datasets, repeat_times, world_size, order, build_context.seed)

    max_batch_size = adjust_max_batch_size(scheduler.get_number_of_tasks(), max_batch_size, world_size, number_of_workers)
    build_context.variables['batch_size'] = max_batch_size

    if is_rank_0_process():
        dynamic_scheduling_server = DistributedTrackerEvaluationTaskDynamicSchedulerServer(
            sampling_orchestrator_server_address, key, scheduler, max_batch_size, initialize=not persistent)
        if persistent:
            build_context.services.event.register_on_start_event_listener(
                lambda: dynamic_scheduling_server.launch(), priority=0)
            build_context.services.event.register_on_stop_event_listener(
                lambda: dynamic_scheduling_server.stop(wait_for_stop=True), priority=99)
            reset_signal_client = DistributedTrackerEvaluationTaskDynamicSchedulerClient(sampling_orchestrator_server_address, key)
            build_context.services.event.register_on_epoch_begin_event_listener(
                lambda epoch, is_train: reset_signal_client.reset(), priority=0)
            build_context.services.event.register_on_stop_event_listener(lambda: reset_signal_client.close(), priority=0)
        else:
            build_context.services.event.register_on_epoch_begin_event_listener(
                lambda epoch, is_train: dynamic_scheduling_server.launch(), priority=0)
            build_context.services.event.register_on_epoch_end_event_listener(
                lambda epoch, is_train: dynamic_scheduling_server.stop(wait_for_stop=True), priority=99)

    dynamic_scheduling_client = DistributedTrackerEvaluationTaskDynamicSchedulerClient(sampling_orchestrator_server_address, key)

    build_context.variables['number_of_evaluation_tasks'] = scheduler.get_number_of_tasks()
    build_context.variables['evaluation_task_desc'] = _build_eval_task_context_variable(datasets, repeat_times)
    build_context.variables['number_of_evaluation_frames'] = scheduler.get_number_of_frames()

    _print_summary_stat(build_context.variables['number_of_evaluation_tasks'],
                        build_context.variables['number_of_evaluation_frames'],
                        build_context.variables['batch_size'])

    return dynamic_scheduling_client
