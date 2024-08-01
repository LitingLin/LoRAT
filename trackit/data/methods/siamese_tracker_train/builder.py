from tabulate import tabulate
from trackit.core.runtime.build_context import BuildContext
from trackit.data.source.builder import build_data_source
from trackit.data.sampling.per_sequence.builder import build_per_sequence_sampler
from trackit.data.utils.dataloader import build_dataloader
from trackit.data import DataPipeline
from trackit.miscellanies.torch.distributed import get_world_size
from .worker import SiameseTrackerTrainingDataWorker, SiameseTrackerTrainingDataCollator, SiameseTrackerTrainingHostLoggingHook
from .siamese_training_pair_sampling.builder import build_SiamFC_training_pair_sampler
from .transform.builder import build_transform


def build_siamese_tracker_train_data_pipeline(data_config: dict, build_context: BuildContext, config: dict) -> DataPipeline:
    datasets = build_data_source(data_config['source'])

    sampler, datasets_sampling_weight = build_per_sequence_sampler(datasets, data_config['sampler'], build_context)

    num_samples_per_epoch = build_context.variables['num_samples_per_epoch']
    if 'global_batch_size' in data_config:
        batch_size = data_config['global_batch_size']
        if get_world_size() > 1:
            local_batch_size = batch_size // get_world_size()
            if local_batch_size == 0:
                local_batch_size = 1
            print(f'local batch_size is adjusted to {local_batch_size} due to distributed training')
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = data_config['batch_size']

    assert local_batch_size * get_world_size() <= num_samples_per_epoch, \
        f'global batch_size ({local_batch_size * get_world_size()}) should be less than or equal to num_samples_per_epoch ({num_samples_per_epoch})'
    build_context.variables['batch_size'] = local_batch_size

    _print_data_source_stat(datasets, local_batch_size, get_world_size(), num_samples_per_epoch,
                            datasets_sampling_weight)

    siamese_training_pair_sampler = build_SiamFC_training_pair_sampler(datasets, datasets_sampling_weight, sampler,
                                                                       data_config['siamese_training_pair_sampling'])

    transform, transform_batch_collator, transform_host_data_pipelines = build_transform(data_config, config,
                                                                                         build_context)

    num_io_threads = data_config['num_io_threads']
    num_workers = data_config['num_workers']

    worker = SiameseTrackerTrainingDataWorker(datasets, num_samples_per_epoch, local_batch_size,
                                              siamese_training_pair_sampler,
                                              transform, num_io_threads)

    dataloader = build_dataloader(worker, local_batch_size, num_workers, build_context, do_shuffle=False,
                                  collate_fn=SiameseTrackerTrainingDataCollator(transform_batch_collator))
    num_iterations_per_epoch = num_samples_per_epoch // (local_batch_size * get_world_size())
    assert num_iterations_per_epoch == len(dataloader)

    build_context.variables['num_iterations_per_epoch'] = num_iterations_per_epoch

    logging_hook = SiameseTrackerTrainingHostLoggingHook(num_io_threads)
    build_context.services.event.register_on_epoch_begin_event_listener(lambda epoch, is_train: logging_hook.on_epoch_begin())

    _print_data_loader_stat(num_iterations_per_epoch, num_workers, num_io_threads)

    return DataPipeline(dataloader, (logging_hook, *transform_host_data_pipelines))


def _print_data_source_stat(datasets, local_batch_size, num_ranks, num_samples_per_epoch,
                            sampling_weight):
    string = '\t'.join((f'batch_size (global): {local_batch_size * num_ranks}',
                        f'batch_size (local): {local_batch_size}',
                        f'num_samples_per_epoch: {num_samples_per_epoch}'))
    string += '\n'
    string += tabulate(((dataset.get_full_name(), len(dataset), weight) for dataset, weight in zip(datasets, sampling_weight)), headers=('dataset', 'num_sequence', 'weight'))

    print(string)


def _print_data_loader_stat(num_iterations_per_epoch, num_workers, num_io_threads):
    string = '\t'.join((f'num_iterations_per_epoch: {num_iterations_per_epoch}',
                        f'num_workers: {num_workers}',
                        f'num_io_threads: {num_io_threads}'))
    print(string)
