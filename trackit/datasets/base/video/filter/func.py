from trackit.datasets.common.filter.data_cleaning.object_category import DataCleaning_ObjectCategory
from trackit.datasets.common.filter.index_selector import IndexSelector
from trackit.datasets.common.filter.range_selector import RangeSelector
from trackit.datasets.common.filter.proportional_range_selector import ProportionalRangeSelector
from trackit.datasets.common.filter.random_shuffle import RandomShuffle
from trackit.datasets.common.filter.sort_by_sequence_frame_size import SortBySequenceFrameSize
from trackit.datasets.common.filter.data_cleaning.integrity import DataCleaning_Integrity
from trackit.datasets.common.filter.data_cleaning.bounding_box import DataCleaning_BoundingBox
from trackit.datasets.common.filter.data_cleaning.annotation_standard import DataCleaning_AnnotationStandard
from trackit.datasets.common.filter.length_limit import LengthLimit
from .tweak_tool import VideoDatasetTweakTool

__all__ = ['apply_filters_on_video_dataset_']


def apply_filters_on_video_dataset_(dataset: dict, filters: list):
    if len(filters) == 0:
        return dataset

    if 'filters' not in dataset:
        dataset['filters'] = []

    filters_serialized = dataset['filters']

    dataset_tweak_tool = VideoDatasetTweakTool(dataset)

    for filter_ in filters:
        if isinstance(filter_, IndexSelector):
            dataset_tweak_tool.apply_index_filter(filter_.indices)
        elif isinstance(filter_, RangeSelector):
            dataset_tweak_tool.apply_range_selector(filter_.start, filter_.stop, filter_.step)
        elif isinstance(filter_, ProportionalRangeSelector):
            dataset_tweak_tool.apply_proportional_range_selector(filter_.start, filter_.stop)
        elif isinstance(filter_, RandomShuffle):
            dataset_tweak_tool.random_shuffle(filter_.seed)
        elif isinstance(filter_, DataCleaning_BoundingBox):
            if filter_.fit_in_image_size:
                dataset_tweak_tool.bounding_box_fit_in_image_size()
            if filter_.update_validity:
                dataset_tweak_tool.bounding_box_update_validity()
            if filter_.remove_invalid_objects:
                dataset_tweak_tool.bounding_box_remove_non_validity_objects()
            if filter_.remove_empty_objects:
                dataset_tweak_tool.bounding_box_remove_empty_annotation_objects()
        elif isinstance(filter_, DataCleaning_Integrity):
            if filter_.remove_zero_annotation_objects:
                dataset_tweak_tool.remove_zero_annotation_objects()
            if filter_.remove_zero_annotation_video_head_tail:
                dataset_tweak_tool.remove_empty_annotation_head_tail()
            if filter_.remove_invalid_image:
                dataset_tweak_tool.remove_invalid_image()
        elif isinstance(filter_, DataCleaning_ObjectCategory):
            if filter_.category_ids_to_remove is not None:
                dataset_tweak_tool.remove_category_ids(filter_.category_ids_to_remove)
            if filter_.make_category_id_sequential:
                dataset_tweak_tool.make_category_id_sequential()
        elif isinstance(filter_, SortBySequenceFrameSize):
            dataset_tweak_tool.sort_by_sequence_frame_size(filter_.descending)
        elif isinstance(filter_, DataCleaning_AnnotationStandard):
            dataset_tweak_tool.annotation_standard_conversion(filter_.bounding_box_format,
                                                              filter_.bounding_box_coordinate_system)
        elif isinstance(filter_, LengthLimit):
            dataset_tweak_tool.length_limit(filter_.value)
        else:
            raise RuntimeError(f"{type(filter_)} not implemented for Video Dataset")

        filters_serialized.append(filter_.serialize())
