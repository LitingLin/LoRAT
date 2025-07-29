import torch

from trackit.core.runtime.build_context import BuildContext
from trackit.data import DataPipeline
from trackit.data.components.result_collector.builder import build_evaluation_result_collector
from trackit.data.source.builder import build_data_source
from trackit.data.sampling.per_frame.distributed_dynamic_eval_task_scheduling.builder import build_distributed_tracker_evaluation_task_dynamic_scheduler
from trackit.data.utils.dataloader import build_dataloader
from .orchestration import DistributedTrackerEvaluationProgressOrchestrationAndMonitoring
from .worker import SiameseTrackEvaluationDataInputWorker, SiameseTrackEvaluationMainProcessLoggingHook
from .transform.builder import build_data_transform


def build_siamese_tracker_eval_data_pipeline(data_config: dict, build_context: BuildContext, config: dict,
                                             dtype: torch.dtype) -> DataPipeline:
    datasets = build_data_source(data_config['source'])

    num_io_threads = data_config['num_io_threads']
    num_workers = data_config['num_workers']

    print('num_workers:', num_workers, '\tnum_io_threads:', num_io_threads)

    sampler = build_distributed_tracker_evaluation_task_dynamic_scheduler(
        datasets, data_config['sampler'], data_config['batch_size'],
        num_workers if num_workers > 0 else 1, build_context)

    data_processor = build_data_transform(data_config['transform'], config, dtype=dtype)

    worker = SiameseTrackEvaluationDataInputWorker(datasets, sampler, data_processor, num_io_threads)
    data_loader = build_dataloader(worker, batch_size=None,
                                   num_workers=num_workers, build_context=build_context,
                                   do_shuffle=False, collate_fn=None)
    data_loader_with_distributed_orchestration = DistributedTrackerEvaluationProgressOrchestrationAndMonitoring(
        data_loader, num_workers,
        build_context.variables['number_of_evaluation_tasks'], build_context.variables['number_of_evaluation_frames'])

    build_context.services.batch_collective_communication.all_gather.register(
        data_loader_with_distributed_orchestration.all_gather_begin,
        data_loader_with_distributed_orchestration.all_gather_end)
    build_context.services.event.register_on_epoch_begin_event_listener(
        lambda epoch, is_train: data_loader_with_distributed_orchestration.on_epoch_begin())
    build_context.services.event.register_on_epoch_end_event_listener(
        lambda epoch, is_train: data_loader_with_distributed_orchestration.on_epoch_end())

    build_context.variables['num_workers'] = num_workers if num_workers > 0 else 1

    logging_hook = SiameseTrackEvaluationMainProcessLoggingHook(num_io_threads)
    build_context.services.event.register_on_epoch_begin_event_listener(
        lambda epoch, is_train: logging_hook.on_epoch_begin())
    data_pipelines_on_main_process = [logging_hook]
    if 'result_collector' in data_config:
        print('tracking result collector is enabled.')
        data_pipelines_on_main_process.append(build_evaluation_result_collector(data_config['result_collector'], build_context))

    return DataPipeline(data_loader_with_distributed_orchestration, data_pipelines_on_main_process)


def build_siamese_tracker_eval_vot_integrated_data_pipeline(data_config: dict, build_context: BuildContext, config: dict,
                                                            dtype: torch.dtype) -> DataPipeline:
    transform = build_data_transform(data_config['transform'], config, build_context.device, dtype)

    import trax
    from trackit.core.third_party.vot.vot_integration import VOT
    from .vot_integration import SiameseTrackerEvaluation_VOTToolkitIntegrator

    vot_region_format = data_config['region_format']
    if vot_region_format == 'mask':
        vot_region_format = trax.Region.MASK
    elif vot_region_format == 'rectangle':
        vot_region_format = trax.Region.RECTANGLE
    else:
        raise ValueError(f"Invalid VOT region format: {vot_region_format}, expected 'rectangle' or 'mask'")
    vot = VOT(vot_region_format, multiobject=True)
    integrator = SiameseTrackerEvaluation_VOTToolkitIntegrator(vot, vot_region_format, transform)

    build_context.variables['num_workers'] = 1
    build_context.variables['batch_size'] = len(vot.objects())

    return DataPipeline(integrator, (integrator,))
