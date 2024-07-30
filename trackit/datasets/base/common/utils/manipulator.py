from trackit.datasets.base.operator.bbox.utility.image import bounding_box_fit_in_image_boundary
from trackit.datasets.base.operator.bbox.utility.image import bounding_box_is_intersect_with_image
from trackit.datasets.base.operator.bbox.transform.compile import compile_bbox_transform
from trackit.datasets.common.types.bounding_box import BoundingBoxCoordinateSystem


def fit_objects_bounding_box_in_image_size(frame, context, exclude_non_validity=True):
    image_size = frame.get_image_size()
    for object_ in frame:
        if object_.has_bounding_box():
            bounding_box, bounding_box_validity = object_.get_bounding_box()
            if exclude_non_validity:
                if bounding_box_validity == False:
                    continue
            bounding_box = bounding_box_fit_in_image_boundary(bounding_box, image_size,
                                                              context.get_bounding_box_format(),
                                                              context.get_bounding_box_coordinate_system())
            object_.set_bounding_box(bounding_box, bounding_box_validity)


def update_objects_bounding_box_validity(frame, context, skip_if_mark_non_validity=True):
    image_size = frame.get_image_size()
    for object_ in frame:
        if object_.has_bounding_box():
            bounding_box, bounding_box_validity = object_.get_bounding_box()
            if skip_if_mark_non_validity:
                if bounding_box_validity == False:
                    continue
            bounding_box_validity = bounding_box_is_intersect_with_image(bounding_box, image_size,
                                                                         context.get_bounding_box_format(),
                                                                         context.get_bounding_box_coordinate_system())
            object_.set_bounding_box(bounding_box, bounding_box_validity)


def prepare_bounding_box_annotation_standard_conversion(bounding_box_format, bounding_box_coordinate_system,
                                                        context):
    if not context.has_context():
        return None

    if bounding_box_format is None:
        bounding_box_format = context.get_bounding_box_format()
    if bounding_box_coordinate_system is None:
        bounding_box_coordinate_system = context.get_bounding_box_coordinate_system()

    converter = compile_bbox_transform(context.get_bounding_box_format(), bounding_box_format,
                                       context.get_bounding_box_coordinate_system(),
                                       bounding_box_coordinate_system)
    context.set_bounding_box_format(bounding_box_format)
    context.set_bounding_box_coordinate_system(bounding_box_coordinate_system)
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        context.set_bounding_box_data_type(int)
    elif bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous:
        context.set_bounding_box_data_type(float)
    else:
        raise NotImplementedError
    return converter
