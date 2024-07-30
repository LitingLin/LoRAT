from .vot_integration import VOT, Rectangle
from torchvision.io.image import read_image, ImageReadMode


def _bbox_convert_from_vot(bbox: Rectangle):
    return bbox.x, bbox.y, bbox.width + bbox.x, bbox.height + bbox.y


def _bbox_convert_to_vot(bbox):
    return Rectangle(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


def _decode_image(path):
    return read_image(path, ImageReadMode.RGB)


def run_tracker(tracker):
    vot_object = VOT("rectangle")
    initial_bbox = _bbox_convert_from_vot(vot_object.region())
    initial_frame_path = vot_object.frame()
    assert isinstance(initial_frame_path, str)
    initial_frame = _decode_image(initial_frame_path)
    tracker.initialize(initial_frame, initial_bbox)
    while True:
        next_frame_path = vot_object.frame()
        if next_frame_path is None:
            break
        assert isinstance(next_frame_path, str)
        predicted_bbox, confidence_score = tracker.track(_decode_image(next_frame_path))
        vot_object.report(_bbox_convert_to_vot(predicted_bbox), confidence_score)
