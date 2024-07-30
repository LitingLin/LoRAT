from trackit.datasets.base.operator.bbox.continuous.iou import bbox_compute_iou


def parse_THOTH_bb_file(file: str):
    frame_annotations = {}
    object_id_last_bounding_box = {}
    for line in open(file):
        line = line.strip()
        if len(line) == 0:
            continue
        words = line.split()
        if len(words) == 1:
            continue
        index = 1
        annotation = {}
        frame_index = int(words[0])
        while True:
            if index >= len(words):
                break
            bounding_box = [float(words[index]), float(words[index + 1]), float(words[index + 2]), float(words[index + 3])]
            confidence = float(words[index + 4])
            index += 5
            object_id = None
            max_iou = 0
            for candidate_object_id, candidate_bounding_box in object_id_last_bounding_box.items():
                iou = bbox_compute_iou(bounding_box, candidate_bounding_box)
                if candidate_object_id in annotation:
                    continue
                if iou > max_iou:
                    max_iou = iou
                    object_id = candidate_object_id
            if object_id is None:
                object_id = len(object_id_last_bounding_box)
            annotation[object_id] = (bounding_box, confidence)
        for object_id, (bounding_box, _) in annotation.items():
            object_id_last_bounding_box[object_id] = bounding_box

        frame_annotations[frame_index] = annotation

    return frame_annotations
