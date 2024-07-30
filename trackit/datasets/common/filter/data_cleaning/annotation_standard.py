from typing import Optional, Union
from .._common import _BaseFilter
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat, BoundingBoxCoordinateSystem


class DataCleaning_AnnotationStandard(_BaseFilter):
    def __init__(self, bounding_box_format: Optional[Union[str, BoundingBoxFormat]] = None,
                 bounding_box_coordinate_system: Optional[Union[str, BoundingBoxCoordinateSystem]] = None):
        if isinstance(bounding_box_format, str):
            self.bounding_box_format = BoundingBoxFormat[bounding_box_format]
        else:
            self.bounding_box_format = bounding_box_format

        if isinstance(bounding_box_coordinate_system, str):
            self.bounding_box_coordinate_system = BoundingBoxCoordinateSystem[bounding_box_coordinate_system]
        else:
            self.bounding_box_coordinate_system = bounding_box_coordinate_system

    def serialize(self):
        return self.__class__.__name__, {
            'bounding_box_format': self.bounding_box_format.name,
            'bounding_box_coordinate_system': self.bounding_box_coordinate_system.name
        }
