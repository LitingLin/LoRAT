from trackit.datasets.common.factory import DatasetFactory
from trackit.datasets.base.image.dataset import ImageDataset
from trackit.datasets.base.image.filter.func import apply_filters_on_image_dataset_
from .constructor import DetectionDatasetConstructorGenerator
from .specialization.memory_mapped.dataset import DetectionDataset_MemoryMapped
from .specialization.memory_mapped.constructor import construct_detection_dataset_memory_mapped_from_base_image_dataset
from typing import List

__all__ = ['DetectionDatasetFactory']


class DetectionDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(DetectionDatasetFactory, self).__init__(seeds, ImageDataset,
                                                      DetectionDatasetConstructorGenerator,
                                                      apply_filters_on_image_dataset_,
                                                      DetectionDataset_MemoryMapped,
                                                      construct_detection_dataset_memory_mapped_from_base_image_dataset)

    def construct(self, filters: list=None, cache_base_format: bool=True, dump_human_readable: bool=False) -> List[DetectionDataset_MemoryMapped]:
        return super(DetectionDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[ImageDataset]:
        return super(DetectionDatasetFactory, self).construct_base_interface(filters, make_cache, dump_human_readable)
