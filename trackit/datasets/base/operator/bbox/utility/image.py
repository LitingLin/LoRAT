from trackit.datasets.common.types.bounding_box import BoundingBoxFormat, BoundingBoxCoordinateSystem


def _common_routine(bounding_box, image_size, bounding_box_format: BoundingBoxFormat,
                    bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                    coord_discrete_xyxy_func, coord_discrete_polygon_func,
                    coord_continuous_xyxy_func, coord_continuous_polygon_func):
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        if bounding_box_format == BoundingBoxFormat.XYWH or bounding_box_format == BoundingBoxFormat.XYXY:
            if bounding_box_format == BoundingBoxFormat.XYWH:
                from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
                bounding_box = bbox_xywh2xyxy(bounding_box)
            return coord_discrete_xyxy_func(bounding_box, image_size)
        else:
            return coord_discrete_polygon_func(bounding_box, image_size)
    else:
        if bounding_box_format == BoundingBoxFormat.XYWH or bounding_box_format == BoundingBoxFormat.XYXY:
            if bounding_box_format == BoundingBoxFormat.XYWH:
                from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
                bounding_box = bbox_xywh2xyxy(bounding_box)
            return coord_continuous_xyxy_func(bounding_box, image_size)
        else:
            return coord_continuous_polygon_func(bounding_box, image_size)


def bounding_box_is_intersect_with_image(bounding_box, image_size, bounding_box_format: BoundingBoxFormat,
                                         bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    import trackit.datasets.base.operator.bbox.discrete.utility.image as discrete_image_ops
    import trackit.datasets.base.operator.bbox.continuous.utility.image as continuous_image_ops
    return _common_routine(bounding_box, image_size, bounding_box_format,
                           bounding_box_coordinate_system,
                           discrete_image_ops.bounding_box_is_intersect_with_image,
                           discrete_image_ops.bounding_box_is_intersect_with_image_polygon,
                           continuous_image_ops.bounding_box_is_intersect_with_image,
                           continuous_image_ops.bounding_box_is_intersect_with_image_polygon_format)


def bounding_box_fit_in_image_boundary(bounding_box, image_size, bounding_box_format: BoundingBoxFormat,
                                       bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    import trackit.datasets.base.operator.bbox.discrete.utility.image as discrete_image_ops
    import trackit.datasets.base.operator.bbox.continuous.utility.image as continuous_image_ops
    return _common_routine(bounding_box, image_size, bounding_box_format,
                           bounding_box_coordinate_system,
                           discrete_image_ops.bounding_box_fit_in_image_boundary,
                           discrete_image_ops.bounding_box_fit_in_image_boundary_polygon,
                           continuous_image_ops.bounding_box_fit_in_image_boundary,
                           continuous_image_ops.bounding_box_fit_in_image_boundary_polygon)
