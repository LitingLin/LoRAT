def bbox_polygon2xyxy(bbox):
    xs = bbox[0::2]
    ys = bbox[1::2]

    bbox = min(xs), min(ys), max(xs), max(ys)
    return bbox
