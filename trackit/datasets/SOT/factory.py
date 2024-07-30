from typing import List, Iterable, Optional
from trackit.datasets.common.factory import DatasetFactory
from trackit.datasets.base.video.dataset import VideoDataset
from trackit.datasets.base.video.filter.func import apply_filters_on_video_dataset_
from .constructor import SingleObjectTrackingDatasetConstructorGenerator
from .specialization.memory_mapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from .specialization.memory_mapped.constructor import construct_single_object_tracking_dataset_memory_mapped_from_base_video_dataset

__all__ = ['SingleObjectTrackingDatasetFactory']


class SingleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: Iterable):
        super(SingleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                 SingleObjectTrackingDatasetConstructorGenerator,
                                                                 apply_filters_on_video_dataset_,
                                                                 SingleObjectTrackingDataset_MemoryMapped,
                                                                 construct_single_object_tracking_dataset_memory_mapped_from_base_video_dataset)

    def construct(self, filters: Optional[Iterable]=None, cache_base_format: bool=True, dump_human_readable: bool=False) -> List[SingleObjectTrackingDataset_MemoryMapped]:
        return super(SingleObjectTrackingDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(SingleObjectTrackingDatasetFactory, self).construct_base_interface(filters, make_cache, dump_human_readable)
