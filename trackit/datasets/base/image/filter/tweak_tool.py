from trackit.datasets.base.image.manipulator import ImageDatasetManipulator
import numpy as np
import copy
from typing import Optional, Sequence, Iterable
from trackit.datasets.base.common.utils.manipulator import fit_objects_bounding_box_in_image_size, \
    update_objects_bounding_box_validity, prepare_bounding_box_annotation_standard_conversion
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat, BoundingBoxCoordinateSystem


class ImageDatasetTweakTool:
    def __init__(self, dataset: dict):
        self.manipulator = ImageDatasetManipulator(dataset)

    def apply_index_filter(self, indices: Iterable[int]):
        self.manipulator.apply_element_selector(indices)

    def apply_range_selector(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None):
        self.manipulator.apply_element_selector(slice(start, stop, step))

    def apply_proportional_range_selector(self, start: Optional[float] = None, stop: Optional[float] = None):
        total_size = len(self.manipulator)
        if start is not None:
            start = int(start * total_size)
        if stop is not None:
            stop = int(stop * total_size)
        self.manipulator.apply_element_selector(slice(start, stop))

    def random_shuffle(self, seed: Optional[int]):
        indices = np.arange(len(self.manipulator))
        rng_engine = np.random.default_rng(seed)
        rng_engine.shuffle(indices)
        self.manipulator.apply_element_selector(indices)

    def sort_by_image_size_ratio(self, descending=False):
        image_sizes = []
        for image in self.manipulator:
            image_sizes.append(image.get_image_size())
        image_sizes = np.array(image_sizes)
        if descending:
            ratio = image_sizes[:, 0] / image_sizes[:, 1]
        else:
            ratio = image_sizes[:, 1] / image_sizes[:, 0]
        indices = ratio.argsort()
        self.manipulator.apply_index_filter(indices)

    def bounding_box_fit_in_image_size(self, exclude_non_validity=True):
        for image in self.manipulator:
            fit_objects_bounding_box_in_image_size(image, self.manipulator.context_dao, exclude_non_validity)

    def bounding_box_update_validity(self, skip_if_mark_non_validity=True):
        for image in self.manipulator:
            update_objects_bounding_box_validity(image, self.manipulator.context_dao, skip_if_mark_non_validity)

    def bounding_box_remove_non_validity_objects(self):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_bounding_box():
                    _, validity = object_.get_bounding_box()
                    if validity == False:
                        object_.delete()

    def annotation_standard_conversion(self, bounding_box_format: Optional[BoundingBoxFormat] = None,
                                       bounding_box_coordinate_system: Optional[BoundingBoxCoordinateSystem] = None):
        converter = prepare_bounding_box_annotation_standard_conversion(bounding_box_format,
                                                                        bounding_box_coordinate_system,
                                                                        self.manipulator.context_dao)
        if converter is None:
            return

        for image in self.manipulator:
            for object_ in image:
                if object_.has_bounding_box():
                    bounding_box, bounding_box_validity = object_.get_bounding_box()
                    bounding_box = converter(bounding_box)
                    object_.set_bounding_box(bounding_box, bounding_box_validity)

    def bounding_box_remove_empty_annotation_objects(self):
        for image in self.manipulator:
            for object_ in image:
                if not object_.has_bounding_box():
                    object_.delete()

    def remove_empty_annotation(self):
        for image in self.manipulator:
            if len(image) == 0:
                image.delete()

    def remove_invalid_image(self):
        for image in self.manipulator:
            w, h = image.get_image_size()
            if w == 0 or h == 0:
                image.delete()

    def remove_category_ids(self, category_ids: Sequence[int]):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_category_id():
                    if object_.get_category_id() in category_ids:
                        object_.delete()

        category_id_name_map: dict = copy.copy(self.manipulator.get_category_id_name_map())
        for category_id in category_ids:
            category_id_name_map.pop(category_id)
        self.manipulator.set_category_id_name_map(category_id_name_map)

    def make_category_id_sequential(self):
        category_id_name_map = self.manipulator.get_category_id_name_map()
        new_category_ids = list(range(len(category_id_name_map)))
        old_new_category_id_map = {o: n for n, o in zip(new_category_ids, category_id_name_map.keys())}
        for image in self.manipulator:
            for object_ in image:
                if object_.has_category_id():
                    if object_.get_category_id() in old_new_category_id_map:
                        object_.set_category_id(old_new_category_id_map[object_.get_category_id()])
        new_category_id_name_map = {n: category_id_name_map[o] for n, o in
                                    zip(new_category_ids, category_id_name_map.keys())}
        self.manipulator.set_category_id_name_map(new_category_id_name_map)