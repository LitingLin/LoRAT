import numpy as np

import trackit.datasets.CLS.specialization.memory_mapped.dataset as CLS_dataset
from trackit.datasets.common.specialization.memory_mapped.constructor import memory_mapped_constructor_common_preliminary_works, \
    memory_mapped_constructor_commit_data
from trackit.datasets.base.common.constructor import image_dataset_key_exclude_list, image_dataset_image_key_exclude_list


def construct_image_classification_dataset_memory_mapped_from_base_image_dataset(base_dataset: dict, path: str):
    constructor, bounding_box_data_type = memory_mapped_constructor_common_preliminary_works(base_dataset, 'image',
                                                                                             path,
                                                                                             CLS_dataset.__version__,
                                                                                             'ImageClassification',
                                                                                             image_dataset_key_exclude_list)
    images_list = []

    for base_image in base_dataset['images']:
        image_attributes = {
            'path': base_image['path'],
            'size': np.asarray(base_image['size']),
            'category_id': base_image['category_id']
        }

        optional_image_attributes = {}

        for base_image_key, base_image_value in base_image.items():
            if base_image_key in image_dataset_image_key_exclude_list:
                continue
            optional_image_attributes[base_image_key] = base_image_value

        if len(optional_image_attributes) == 0:
            optional_image_attributes = None

        images_list.append((image_attributes, optional_image_attributes))

    return memory_mapped_constructor_commit_data(images_list, constructor)
