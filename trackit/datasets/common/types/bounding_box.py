import enum


class BoundingBoxCoordinateSystem(enum.Enum):
    Discrete = enum.auto() # index of pixels
    Continuous = enum.auto()


class BoundingBoxFormat(enum.Enum):
    r'''
    XYWH: [x, y, w, h]
    XYXY: [x1, y1, x2, y2]
    Polygon: [x1, y1, x2, y2, x3, y3, x4, y4]
    '''
    XYWH = enum.auto()
    XYXY = enum.auto()
    Polygon = enum.auto()
