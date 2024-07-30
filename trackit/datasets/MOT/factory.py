from trackit.datasets.common.factory import DatasetFactory
from trackit.datasets.base.video.dataset import VideoDataset
from trackit.datasets.base.video.filter.func import apply_filters_on_video_dataset_
from .constructor import MultipleObjectTrackingDatasetConstructorGenerator
from .specialization.memory_mapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from .specialization.memory_mapped.constructor import construct_multiple_object_tracking_dataset_memory_mapped_from_base_video_dataset
from typing import List, Iterable, Optional

__all__ = ['MultipleObjectTrackingDatasetFactory']


class MultipleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(MultipleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                   MultipleObjectTrackingDatasetConstructorGenerator,
                                                                   apply_filters_on_video_dataset_,
                                                                   MultipleObjectTrackingDataset_MemoryMapped,
                                                                   construct_multiple_object_tracking_dataset_memory_mapped_from_base_video_dataset)

    def construct(self, filters: Optional[Iterable] = None, cache_base_format: bool=True, dump_human_readable: bool=False) -> List[MultipleObjectTrackingDataset_MemoryMapped]:
        return super(MultipleObjectTrackingDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(MultipleObjectTrackingDatasetFactory, self).construct_base_interface(filters, make_cache, dump_human_readable)
