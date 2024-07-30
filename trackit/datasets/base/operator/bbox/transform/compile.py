from trackit.datasets.common.types.bounding_box import BoundingBoxFormat, BoundingBoxCoordinateSystem


def _compile_bbox_xyxy_transform(commands: list,
                                 source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                 target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    from .coord import bbox_coord_continuous_to_discrete, bbox_coord_discrete_to_continuous
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous:
        commands.append(bbox_coord_discrete_to_continuous)
    elif source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        commands.append(bbox_coord_continuous_to_discrete)


def _compile_bbox_polygon_transform(commands: list,
                                    source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                    target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    from .coord import bbox_coord_discrete_to_continuous_polygon, bbox_coord_continuous_to_discrete_polygon
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous:
        commands.append(bbox_coord_discrete_to_continuous_polygon)
    elif source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Continuous and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        commands.append(bbox_coord_continuous_to_discrete_polygon)


def _compile_bbox_rasterized_transform(commands: list, source_format: BoundingBoxFormat,
                                       target_format: BoundingBoxFormat):
    if source_format == target_format:
        return
    if source_format == BoundingBoxFormat.XYWH:
        from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
        commands.append(bbox_xywh2xyxy)
    elif source_format == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)
    if target_format == BoundingBoxFormat.XYWH:
        from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
        commands.append(bbox_xyxy2xywh)
    elif target_format == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        commands.append(bbox_xyxy2polygon)


def _compile_bbox_to_xyxy(commands: list, format_: BoundingBoxFormat,
                          bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYXY:
        return
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
            from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)
        else:
            from trackit.datasets.base.operator.bbox.continuous.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)
    elif format_ == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)


def _compile_bbox_to_polygon(commands: list, format_: BoundingBoxFormat,
                             bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.Polygon:
        return
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
            from trackit.datasets.base.operator.bbox.discrete.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)
        else:
            from trackit.datasets.base.operator.bbox.continuous.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)

    from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    commands.append(bbox_xyxy2polygon)


def _compile_bbox_polygon_to_any(commands: list, target_format: BoundingBoxFormat,
                                 bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.Polygon:
        return
    else:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)
        if target_format == BoundingBoxFormat.XYWH:
            if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
                from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
                commands.append(bbox_xyxy2xywh)
            else:
                from trackit.datasets.base.operator.bbox.continuous.xyxy2xywh import bbox_xyxy2xywh
                commands.append(bbox_xyxy2xywh)


def _compile_bbox_xyxy_to_any(commands: list, target_format: BoundingBoxFormat,
                              bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.XYXY:
        return
    elif target_format == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
            from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
            commands.append(bbox_xyxy2xywh)
        else:
            from trackit.datasets.base.operator.bbox.continuous.xyxy2xywh import bbox_xyxy2xywh
            commands.append(bbox_xyxy2xywh)
    else:
        from trackit.datasets.base.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        commands.append(bbox_xyxy2polygon)


def _compile_bbox_to_xywh(commands: list, format_: BoundingBoxFormat,
                          bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYWH:
        return
    if format_ == BoundingBoxFormat.Polygon:
        from trackit.datasets.base.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        from trackit.datasets.base.operator.bbox.discrete.xyxy2xywh import bbox_xyxy2xywh
        commands.append(bbox_xyxy2xywh)
    else:
        from trackit.datasets.base.operator.bbox.continuous.xyxy2xywh import bbox_xyxy2xywh
        commands.append(bbox_xyxy2xywh)


def compile_bbox_transform(source_format: BoundingBoxFormat, target_format: BoundingBoxFormat,
                           source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                           target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    commands = []
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Discrete:
        # do in integer space
        _compile_bbox_rasterized_transform(commands, source_format, target_format)
    else:
        # do in float space
        if source_format == BoundingBoxFormat.Polygon or target_format == BoundingBoxFormat.Polygon:
            # do in polygon routine
            _compile_bbox_to_polygon(commands, source_format, source_bounding_box_coordinate_system)
            _compile_bbox_polygon_transform(commands,
                                            source_bounding_box_coordinate_system,
                                            target_bounding_box_coordinate_system)
            _compile_bbox_polygon_to_any(commands, target_format, target_bounding_box_coordinate_system)
        else:
            # do in xyxy routine
            _compile_bbox_to_xyxy(commands, source_format, source_bounding_box_coordinate_system)
            _compile_bbox_xyxy_transform(commands,
                                         source_bounding_box_coordinate_system, target_bounding_box_coordinate_system)
            _compile_bbox_xyxy_to_any(commands, target_format, target_bounding_box_coordinate_system)

    class _BoundingBoxConverter:
        def __init__(self, commands_):
            self.commands = commands_

        def __call__(self, bbox):
            for command in self.commands:
                bbox = command(bbox)
            return bbox
    return _BoundingBoxConverter(commands)
