from trackit.datasets.common.factory import DatasetFactory
from trackit.datasets.base.image.dataset import ImageDataset
from trackit.datasets.base.image.filter.func import apply_filters_on_image_dataset_
from .constructor import ImageClassificationDatasetConstructorGenerator
from .specialization.memory_mapped.dataset import ImageClassificationDataset_MemoryMapped
from .specialization.memory_mapped.constructor import construct_image_classification_dataset_memory_mapped_from_base_image_dataset
from typing import List

__all__ = ['ImageClassificationDatasetFactory']


class ImageClassificationDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(ImageClassificationDatasetFactory, self).__init__(seeds, ImageDataset,
                                                              ImageClassificationDatasetConstructorGenerator,
                                                              apply_filters_on_image_dataset_,
                                                              ImageClassificationDataset_MemoryMapped,
                                                              construct_image_classification_dataset_memory_mapped_from_base_image_dataset)

    def construct(self, filters: list = None, cache_base_format: bool = True, dump_human_readable: bool = False) -> List[ImageClassificationDataset_MemoryMapped]:
        return super(ImageClassificationDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[ImageDataset]:
        return super(ImageClassificationDatasetFactory, self).construct_base_interface(filters, make_cache,
                                                                                       dump_human_readable)
