import os
from trackit.datasets.base.common.dataset import _BaseDataset, _BaseDatasetObject
from trackit.miscellanies.operating_system.path import join_paths

__version__ = 4


class ImageDatasetImage:
    def __init__(self, image: dict, root_path: str):
        self.image = image
        self.root_path = root_path

    def has_category_id(self):
        return 'category_id' in self.image

    def get_category_id(self):
        return self.image['category_id']

    def has_attribute(self, name: str):
        return name in self.image

    def get_attribute(self, name: str):
        return self.image[name]

    def get_all_attribute_name(self):
        return self.image.keys()

    def get_image_file_name(self):
        return os.path.basename(self.image['path'])

    def get_image_path(self):
        return join_paths(self.root_path, self.image['path'])

    def get_image_size(self):
        return self.image['size']

    def __len__(self):
        if 'objects' not in self.image:
            return 0
        return len(self.image['objects'])

    def __getitem__(self, index: int):
        return _BaseDatasetObject(self.image['objects'][index])


class ImageDataset(_BaseDataset):
    def __init__(self, root_path: str, dataset: dict=None):
        super(ImageDataset, self).__init__(root_path, dataset)

    def __getitem__(self, index: int):
        return ImageDatasetImage(self.dataset['images'][index], self.root_path)

    def __len__(self):
        return len(self.dataset['images'])

    @classmethod
    def load(cls, dataset_root_path: str, meta_data_file_path: str):
        return cls(dataset_root_path, _BaseDataset._load(meta_data_file_path, __version__))
