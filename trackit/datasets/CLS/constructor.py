from trackit.datasets.base.common.constructor import BaseDatasetConstructorGenerator, BaseImageDatasetConstructor, \
    BaseDatasetImageConstructorGenerator, BaseDatasetImageConstructor


class ImageClassificationDatasetImageConstructor(BaseDatasetImageConstructor):
    def set_category_id(self, category_id):
        assert self.category_id_name_map is not None, "set set_category_id_name_map first"
        assert category_id in self.category_id_name_map
        self.image['category_id'] = category_id


class ImageClassificationDatasetImageConstructorGenerator(BaseDatasetImageConstructorGenerator):
    def __init__(self, image: dict, root_path: str, category_id_name_map: dict, context):
        super(ImageClassificationDatasetImageConstructorGenerator, self).__init__(image, context)
        self.root_path = root_path
        self.category_id_name_map = category_id_name_map

    def __enter__(self):
        return ImageClassificationDatasetImageConstructor(self.image, self.root_path, self.context, self.category_id_name_map)


class ImageClassificationDatasetConstructor(BaseImageDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, version: int, context):
        super(ImageClassificationDatasetConstructor, self).__init__(dataset, root_path, version, context)

    def new_image(self):
        image = {}
        self.dataset['images'].append(image)
        assert 'category_id_name_map' in self.dataset
        category_id_name_map = self.dataset['category_id_name_map']
        return ImageClassificationDatasetImageConstructorGenerator(image, self.root_path, category_id_name_map, self.context)


class ImageClassificationDatasetConstructorGenerator(BaseDatasetConstructorGenerator):
    def __init__(self, dataset: dict, root_path: str, version: int):
        super(ImageClassificationDatasetConstructorGenerator, self).__init__(dataset, root_path, version,
                                                                             ImageClassificationDatasetConstructor)
