from typing import Tuple, Mapping
from trackit.datasets.common.types.exception import IncompatibleError
from trackit.datasets.common.utils.filter import filter_list_deserialize
from trackit.datasets.common.dataset_context_dao import DatasetContextDAO
from trackit.datasets.common.unique_id import generate_dataset_unique_id
from .engine import ListMemoryMapped


class LazyAttributesLoader:
    def __init__(self, storage, index):
        self.storage = storage
        self.index = index
        self.attributes = None

    def _try_load_attributes(self):
        if self.attributes is None:
            self.attributes = self.storage[self.index]
            assert self.attributes is not None

    def get_attribute(self, key):
        self._try_load_attributes()
        return self.attributes[key]

    def has_attribute(self, key):
        self._try_load_attributes()
        return key in self.attributes

    def get_all_attribute_name(self):
        self._try_load_attributes()
        return self.attributes.keys()

    def get_attribute_tree_query(self, key):
        self._try_load_attributes()
        value = self.attributes
        for param in key:
            value = value[param]
        return value

    def has_attribute_tree_query(self, key):
        self._try_load_attributes()
        value = self.attributes
        for key in key:
            if key not in value:
                return False
            value = value[key]
        return True

    def get_all_attribute_name_tree_query(self, key):
        self._try_load_attributes()
        value = self.get_attribute_tree_query(key)
        return value.keys()


class DummyAttributesLoader:
    def get_attribute(self, _):
        raise KeyError

    def has_attribute(self, _):
        return False

    def get_all_attribute_name(self):
        return None

    def get_attribute_tree_query(self, _):
        raise KeyError

    def has_attribute_tree_query(self, _):
        return False

    def get_all_attribute_name_tree_query(self, _):
        return None


class MemoryMappedDataset:
    def __init__(self, root_path: str, storage: ListMemoryMapped, schema_version, dataset_type_name):
        self.root_path = root_path
        self.storage = storage
        '''
        dataset_attributes: 
            {
                'name': str
                'category_id_name_map': dict[nullable], mapping category_id => category_name, null means no category info
                ...
            }
        '''
        self.dataset_attributes: dict = self.storage[0]
        if self.dataset_attributes['version'][0] != schema_version or self.dataset_attributes['type'] != dataset_type_name:
            del self.dataset_attributes
            del self.storage
            raise IncompatibleError()
        self.index_matrix = self.storage[1].copy()
        self.context = DatasetContextDAO(self.dataset_attributes)

    @staticmethod
    def _load_storage(path: str):
        return ListMemoryMapped(path)

    def has_category_id_name_map(self):
        return 'category_id_name_map' in self.dataset_attributes

    def get_category_id_name_map(self) -> Mapping[int, str]:
        return self.dataset_attributes['category_id_name_map']

    def get_category_name_by_id(self, id_: int) -> str:
        return self.dataset_attributes['category_id_name_map'][id_]

    def get_data_split(self) -> Tuple[str, ...]:
        return self.dataset_attributes['split']

    def get_version(self) -> int:
        return self.dataset_attributes['version'][1]

    def set_root_path(self, root_path: str):
        self.root_path = root_path

    def get_root_path(self):
        return self.root_path

    def get_applied_filter_list(self):
        if 'filters' in self.dataset_attributes:
            return filter_list_deserialize(self.dataset_attributes['filters'])
        return None

    def get_attribute(self, name):
        return self.dataset_attributes[name]

    def has_attribute(self, name):
        return name in self.dataset_attributes

    def get_all_attribute_name(self):
        return self.dataset_attributes.keys()

    def get_name(self):
        return self.dataset_attributes['name']

    def __len__(self):
        return self.index_matrix.shape[0]

    def get_bounding_box_format(self):
        return self.context.get_bounding_box_format()

    def get_bounding_box_coordinate_system(self):
        return self.context.get_bounding_box_coordinate_system()

    def get_bounding_box_data_type(self):
        return self.context.get_bounding_box_data_type()

    def get_full_name(self):
        return generate_dataset_unique_id(self.get_name(), self.get_data_split(), self.get_applied_filter_list(), False)

    def get_unique_id(self):
        return generate_dataset_unique_id(self.get_name(), self.get_data_split(), self.get_applied_filter_list(), True)
