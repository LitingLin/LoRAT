import numpy as np
from typing import Optional
from trackit.miscellanies.operating_system.path import join_paths
from trackit.datasets.common.specialization.memory_mapped.dataset import LazyAttributesLoader, DummyAttributesLoader, MemoryMappedDataset
from trackit.datasets.common.specialization.memory_mapped.engine import ListMemoryMapped

__version__ = 4
__all__ = ['DetectionDataset_MemoryMapped']


class DetectionDatasetObject_MemoryMapped:
    def __init__(self, object_index: int, bounding_box: Optional[np.ndarray],
                 bounding_box_validity_flag: Optional[np.ndarray], object_category_id: Optional[int],
                 image_additional_attributes: LazyAttributesLoader):
        self.object_index = object_index
        self.bounding_box = bounding_box
        self.bounding_box_validity_flag = bounding_box_validity_flag
        self.object_category_id = object_category_id
        self.image_additional_attributes = image_additional_attributes

    def has_bounding_box(self):
        return self.bounding_box is not None

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag is not None

    def has_category_id(self):
        return self.object_category_id is not None

    def get_bounding_box(self):
        return self.bounding_box

    def get_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag

    def get_category_id(self):
        return self.object_category_id

    def get_attribute(self, name: str):
        return self.image_additional_attributes.get_attribute_tree_query(('objects', self.object_index, name))

    def has_attribute(self, name: str):
        return self.image_additional_attributes.has_attribute_tree_query(('objects', self.object_index, name))

    def get_all_attribute_name(self):
        return self.image_additional_attributes.get_all_attribute_name_tree_query(('objects', self.object_index))


class DetectionDatasetImage_MemoryMapped:
    def __init__(self, root_path: str, image_attributes: dict,
                 bounding_box_matrix: Optional[np.ndarray],
                 bounding_box_validity_flag_matrix: Optional[np.ndarray],
                 object_category_id_vector: Optional[np.ndarray],
                 image_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.image_attributes = image_attributes
        self.bounding_box_matrix = bounding_box_matrix
        self.bounding_box_validity_flag_vector = bounding_box_validity_flag_matrix
        self.object_category_id_vector = object_category_id_vector
        self.image_additional_attributes = image_additional_attributes_loader

    def get_image_path(self):
        return join_paths(self.root_path, self.image_attributes['path'])

    def get_image_size(self):
        return self.image_attributes['size']

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        bounding_box = self.bounding_box_matrix[index, :] if self.bounding_box_matrix is not None else None
        bounding_box_validity_flag = self.bounding_box_validity_flag_vector[index] if self.bounding_box_validity_flag_vector is not None else None
        object_category_id = self.object_category_id_vector[index].item() if self.object_category_id_vector is not None else None
        if object_category_id == -1:
            object_category_id = None

        return DetectionDatasetObject_MemoryMapped(index, bounding_box,
                                                   bounding_box_validity_flag, object_category_id,
                                                   self.image_additional_attributes)

    def __len__(self):
        return self.image_attributes['number_of_objects']

    def get_all_bounding_box(self):
        return self.bounding_box_matrix

    def get_all_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector

    def get_all_category_id(self):
        return self.object_category_id_vector

    def has_category_id(self):
        return self.object_category_id_vector is not None

    def has_bounding_box(self):
        return self.bounding_box_matrix is not None

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector is not None

    def has_attribute(self, name: str):
        return self.image_additional_attributes.has_attribute(name)

    def get_attribute(self, name: str):
        return self.image_additional_attributes.get_attribute(name)

    def get_all_attribute_name(self):
        return self.image_additional_attributes.get_all_attribute_name()


class DetectionDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(DetectionDataset_MemoryMapped, self).__init__(root_path, storage, __version__, 'Detection')

    @staticmethod
    def load(root_path: str, path: str):
        return DetectionDataset_MemoryMapped(root_path, MemoryMappedDataset._load_storage(path))

    def __getitem__(self, index: int):
        image_attribute = self.storage[self.index_matrix[index, 0]]

        bounding_box_matrix_index = self.index_matrix[index, 1]
        bounding_box_matrix = self.storage[bounding_box_matrix_index] if bounding_box_matrix_index != -1 else None

        bounding_box_validity_flag_vector_index = self.index_matrix[index, 2]
        bounding_box_validity_flag_vector = self.storage[
            bounding_box_validity_flag_vector_index] if bounding_box_validity_flag_vector_index != -1 else None

        object_category_id_vector_index = self.index_matrix[index, 3]
        object_category_id_vector = self.storage[
            object_category_id_vector_index] if object_category_id_vector_index != -1 else None

        image_additional_attributes_index = self.index_matrix[index, 4]

        if image_additional_attributes_index != -1:
            image_additional_attributes_loader = LazyAttributesLoader(self.storage, image_additional_attributes_index)
        else:
            image_additional_attributes_loader = DummyAttributesLoader()

        return DetectionDatasetImage_MemoryMapped(self.root_path,
                                                  image_attribute, bounding_box_matrix,
                                                  bounding_box_validity_flag_vector,
                                                  object_category_id_vector,
                                                  image_additional_attributes_loader)
