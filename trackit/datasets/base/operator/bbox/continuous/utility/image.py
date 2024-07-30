def get_image_bounding_box(image_size):
    from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
    bbox = bbox_xywh2xyxy((0, 0, image_size[0], image_size[1]))
    return bbox


def bounding_box_is_intersect_with_image(bounding_box, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from trackit.datasets.base.operator.bbox.intersection import bbox_get_intersection
    from trackit.datasets.base.operator.bbox.continuous.validity import bbox_is_valid
    return bbox_is_valid(bbox_get_intersection(image_bounding_box, bounding_box))


def bounding_box_is_intersect_with_image_polygon_format(bounding_box, image_size):
    from trackit.datasets.base.operator.bbox.utility.polygon import get_shapely_polygon_object
    from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    A = get_shapely_polygon_object(bbox_xyxy2polygon(bounding_box))
    image_bounding_box = get_image_bounding_box(image_size)
    B = get_shapely_polygon_object(bbox_xyxy2polygon(image_bounding_box))
    return A.intersection(B).area > 0


def get_image_center_point(image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from trackit.datasets.base.operator.bbox.continuous.center import bbox_get_center_point
    return bbox_get_center_point(image_bounding_box)


def bounding_box_fit_in_image_boundary(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from trackit.datasets.base.operator.bbox.intersection import bbox_fit_in_boundary
    return bbox_fit_in_boundary(bbox, image_bounding_box)


def bounding_box_fit_in_image_boundary_polygon(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from trackit.datasets.base.operator.bbox.intersection import bbox_fit_in_boundary_polygon
    return bbox_fit_in_boundary_polygon(bbox, image_bounding_box)
