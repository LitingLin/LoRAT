def bbox_coord_continuous_to_discrete(bbox, eps=1.0e-4):
    return (int(bbox[0]), int(bbox[1]), int(bbox[2] + 1 - eps), int(bbox[3] + 1 - eps))


def bbox_coord_discrete_to_continuous(bbox):
    return bbox


def bbox_coord_continuous_to_discrete_polygon(bbox):
    return tuple(int(v) for v in bbox)


# note: covering all pixels through rasterization is better, half-pixel center is a tradeoff between performance and accuracy
def bbox_coord_discrete_to_continuous_polygon(bbox):
    return tuple(v + 0.5 for v in bbox)
