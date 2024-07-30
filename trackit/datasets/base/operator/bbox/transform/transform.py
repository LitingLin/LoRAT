from trackit.datasets.common.types.bounding_box import BoundingBoxFormat, BoundingBoxCoordinateSystem


def bbox_xyxy_transform(bbox,
                        source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                        target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    from .coord import bbox_coord_continuous_to_discrete, bbox_coord_discrete_to_continuous
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous:
        return bbox_coord_discrete_to_continuous(bbox)
    elif source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        return bbox_coord_continuous_to_discrete(bbox)
    else:
        return bbox


def bbox_polygon_transform(bbox,
                           source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                           target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    from .coord import bbox_coord_discrete_to_continuous_polygon, bbox_coord_continuous_to_discrete_polygon
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous:
        return bbox_coord_discrete_to_continuous_polygon(bbox)
    elif source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        return bbox_coord_continuous_to_discrete_polygon(bbox)
    else:
        return bbox


def bbox_coord_discrete_transform(bbox, source_format: BoundingBoxFormat, target_format: BoundingBoxFormat):
    if source_format == target_format:
        return bbox
    if source_format == BoundingBoxFormat.XYWH:
        from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
        bbox = bbox_xywh2xyxy(bbox)
    elif source_format == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        bbox = bbox_polygon2xyxy(bbox)
    if target_format == BoundingBoxFormat.XYWH:
        from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
        return bbox_xyxy2xywh(bbox)
    elif target_format == BoundingBoxFormat.XYXY:
        return bbox
    elif target_format == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        return bbox_xyxy2polygon(bbox)


def bbox_to_xyxy(bbox, format_: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYXY:
        return bbox
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
            from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
            return bbox_xywh2xyxy(bbox)
        else:
            from trackit.datasets.base.operator.bbox.continuous.xywh2xyxy import bbox_xywh2xyxy
            return bbox_xywh2xyxy(bbox)
    elif format_ == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        return bbox_polygon2xyxy(bbox)


def bbox_to_polygon(bbox, format_: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.Polygon:
        return bbox
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
            from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
            bbox = bbox_xywh2xyxy(bbox)
        else:
            from trackit.datasets.base.operator.bbox.continuous.xywh2xyxy import bbox_xywh2xyxy
            bbox = bbox_xywh2xyxy(bbox)

    from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    return bbox_xyxy2polygon(bbox)


def bbox_polygon_to_any(bbox, target_format: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.Polygon:
        return bbox
    else:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        bbox = bbox_polygon2xyxy(bbox)
        if target_format == BoundingBoxFormat.XYWH:
            if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
                from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
                return bbox_xyxy2xywh(bbox)
            else:
                from trackit.datasets.base.operator.bbox.continuous.xyxy2xywh import bbox_xyxy2xywh
                return bbox_xyxy2xywh(bbox)
        else:
            return bbox


def bbox_xyxy_to_any(bbox, target_format: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.XYXY:
        return bbox
    elif target_format == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
            from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
            return bbox_xyxy2xywh(bbox)
        else:
            from trackit.datasets.base.operator.bbox.continuous.xyxy2xywh import bbox_xyxy2xywh
            return bbox_xyxy2xywh(bbox)
    else:
        from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        return bbox_xyxy2polygon(bbox)


def bbox_to_xywh(bbox, format_: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYWH:
        return bbox
    if format_ == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        bbox = bbox_polygon2xyxy(bbox)
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
        return bbox_xyxy2xywh(bbox)
    else:
        from trackit.datasets.base.operator.bbox.continuous.xyxy2xywh import bbox_xyxy2xywh
        return bbox_xyxy2xywh(bbox)


def bbox_transform(bbox, source_format: BoundingBoxFormat, target_format: BoundingBoxFormat,
                   source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        # do in integer space
        return bbox_coord_discrete_transform(bbox, source_format, target_format)
    else:
        # do in float space
        if source_format == BoundingBoxFormat.Polygon or target_format == BoundingBoxFormat.Polygon:
            # do in polygon routine
            bbox = bbox_to_polygon(bbox, source_format, source_bounding_box_coordinate_system)
            bbox = bbox_polygon_transform(bbox, source_bounding_box_coordinate_system, target_bounding_box_coordinate_system)
            return bbox_polygon_to_any(bbox, target_format, target_bounding_box_coordinate_system)
        else:
            # do in xyxy routine
            bbox = bbox_to_xyxy(bbox, source_format, source_bounding_box_coordinate_system)
            bbox = bbox_xyxy_transform(bbox, source_bounding_box_coordinate_system, target_bounding_box_coordinate_system)
            return bbox_xyxy_to_any(bbox, target_format, target_bounding_box_coordinate_system)
