import json
import os
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def construct_Youtube_VIS(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    root_path = os.path.join(root_path, 'train')

    with open(os.path.join(root_path, 'instances.json'), 'r', newline='\n') as f:
        annotations = json.load(f)

    constructor.set_category_id_name_map({category_info['id']: category_info['name'] for category_info in annotations['categories']})
    constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)
    images_path = os.path.join(root_path, 'JPEGImages')

    sequences_attributes = {}

    for annotation in annotations['annotations']:
        video_index = annotation['video_id'] - 1
        video_attributes = annotations['videos'][video_index]
        assert video_attributes['id'] == annotation['video_id']
        assert video_attributes['width'] == annotation['width'] and video_attributes['height'] == annotation['height']
        sequence_name = video_attributes['file_names'][0].split('/')[0]
        if sequence_name not in sequences_attributes:
            tracks_bboxes = []
            tracks_category_id = []
            sequences_attributes[sequence_name] = ((video_attributes['width'], video_attributes['height']), video_attributes['file_names'], tracks_bboxes, tracks_category_id)
        else:
            tracks_bboxes = sequences_attributes[sequence_name][2]
            tracks_category_id = sequences_attributes[sequence_name][3]
        tracks_bboxes.append(annotation['bboxes'])
        tracks_category_id.append(annotation['category_id'])

    constructor.set_total_number_of_sequences(len(sequences_attributes))
    for sequence_name, sequence_attributes in sequences_attributes.items():
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for file_name in sequence_attributes[1]:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(images_path, file_name), sequence_attributes[0])
            for index_of_track, (track_bboxes, track_category_id) in enumerate(zip(sequence_attributes[2], sequence_attributes[3])):
                with sequence_constructor.new_object(index_of_track) as object_constructor:
                    object_constructor.set_category_id(track_category_id)
                for index_of_frame, track_bbox in enumerate(track_bboxes):
                    if track_bbox is not None:
                        with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                            with frame_constructor.new_object(index_of_track) as object_constructor:
                                object_constructor.set_bounding_box(track_bbox)
