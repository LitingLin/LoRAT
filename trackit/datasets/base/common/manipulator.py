from trackit.datasets.common.types.bounding_box import BoundingBoxFormat, BoundingBoxCoordinateSystem
from trackit.datasets.common.dataset_context_dao import DatasetContextDAO
from trackit.datasets.base.common.utils.bounding_box import get_bounding_box, set_bounding_box_
from typing import Union, Iterable


class DatasetObjectManipulator:
    def __init__(self, image: dict, index_of_object: int, parent_iterator=None):
        self.object_ = image['objects'][index_of_object]
        self.parent_image = image
        self.index_of_object = index_of_object
        self.parent_iterator = parent_iterator

    def get_id(self):
        return self.object_['id']

    def get_bounding_box(self):
        return get_bounding_box(self.object_)

    def set_bounding_box(self, bounding_box, validity=None):
        set_bounding_box_(self.object_, bounding_box, validity)

    def bounding_box_mark_validity(self, value: bool):
        self.object_['bounding_box']['validity'] = value

    def delete_bounding_box(self):
        del self.object_['bounding_box']

    def get_category_id(self):
        return self.object_['category_id']

    def has_category_id(self):
        return 'category_id' in self.object_

    def has_bounding_box(self):
        return 'bounding_box' in self.object_

    def delete_category_id(self):
        del self.object_['category_id']

    def set_category_id(self, id_: int):
        self.object_['category_id'] = id_

    def get_attribute(self, name: str):
        return self.object_[name]

    def delete(self):
        del self.parent_image['objects'][self.index_of_object]
        del self.object_
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class DatasetObjectManipulatorIterator:
    def __init__(self, frame: dict):
        self.frame = frame
        self.index = 0

    def __next__(self):
        if 'objects' not in self.frame:
            raise StopIteration
        if self.index >= len(self.frame['objects']):
            raise StopIteration

        modifier = DatasetObjectManipulator(self.frame, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class _BaseDatasetManipulator:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.context_dao = DatasetContextDAO(dataset)

    def apply_element_selector(self, selector: Union[Iterable[int], slice]):
        if isinstance(selector, slice):
            self.dataset['sequences'] = self.dataset['sequences'][selector]
        else:
            self.dataset['sequences'] = [self.dataset['sequences'][index] for index in selector]

    def set_name(self, name: str):
        self.dataset['name'] = name

    def get_category_id_name_map(self):
        return self.dataset['category_id_name_map']

    def has_category_id_name_map(self):
        return 'category_id_name_map' in self.dataset

    def set_category_id_name_map(self, category_id_name_map: dict):
        self.dataset['category_id_name_map'] = category_id_name_map

    def _try_allocate_context_object(self):
        if 'context' not in self.dataset:
            self.dataset['context'] = {}

    def set_bounding_box_format(self, bounding_box_format: BoundingBoxFormat):
        self.context_dao.set_bounding_box_format(bounding_box_format)

    def set_bounding_box_coordinate_system(self, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
        self.context_dao.set_bounding_box_coordinate_system(bounding_box_coordinate_system)

    def set_bounding_box_data_type(self, type_):
        self.context_dao.set_bounding_box_data_type(type_)

    def get_bounding_box_format(self):
        return self.context_dao.get_bounding_box_format()

    def get_bounding_box_coordinate_system(self):
        return self.context_dao.get_bounding_box_coordinate_system()

    def get_bounding_box_data_type(self):
        return self.context_dao.get_bounding_box_data_type()
