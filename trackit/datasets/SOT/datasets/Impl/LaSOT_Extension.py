import os
import numpy as np
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


def construct_LaSOT_Extension(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    category_id_name_map = {i: class_name for i, class_name in enumerate(_categories)}
    constructor.set_category_id_name_map(category_id_name_map)
    category_name_id_map = {v: k for k, v in category_id_name_map.items()}

    sequence_names = []
    for category_name in _categories:
        sequence_names.extend([f"{category_name}-{i}" for i in range(1, 11)])

    constructor.set_total_number_of_sequences(len(sequence_names))

    constructor.set_attribute('attribute name', LaSOTExtensionAttributes.ATTRIBUTE_NAMES)
    constructor.set_attribute('attribute short name', LaSOTExtensionAttributes.ATTRIBUTE_SHORT_NAMES)

    for category_name in _categories:
        for sequence_name in [f"{category_name}-{i}" for i in range(1, 11)]:
            category_id = category_name_id_map[category_name]
            class_path = os.path.join(root_path, category_name)
            frame_size = _sequence_frame_size[sequence_name]
            with constructor.new_sequence(category_id) as sequence_constructor:
                sequence_constructor.set_name(sequence_name)
                sequence_constructor.set_fps(30)
                sequence_constructor.set_attribute('attributes',
                                                   LaSOTExtensionAttributes.SEQUENCE_ATTRIBUTES[sequence_name])
                sequence_path = os.path.join(class_path, sequence_name)
                groundtruth_file_path = os.path.join(sequence_path, 'groundtruth.txt')
                bounding_boxes = np.loadtxt(groundtruth_file_path, dtype=np.int64, delimiter=',')
                full_occlusion_file_path = os.path.join(sequence_path, 'full_occlusion.txt')
                is_fully_occluded_flags = np.loadtxt(full_occlusion_file_path, dtype=np.bool_, delimiter=',')
                out_of_view_file_path = os.path.join(sequence_path, 'out_of_view.txt')
                is_out_of_view_flags = np.loadtxt(out_of_view_file_path, dtype=np.bool_, delimiter=',')
                image_folder_path = os.path.join(sequence_path, 'img')
                if len(bounding_boxes) != len(is_fully_occluded_flags) != len(is_out_of_view_flags):
                    raise Exception('annotation length mismatch in {}'.format(sequence_path))

                image_file_names = tuple(f'{i + 1:08d}.jpg' for i in range(len(bounding_boxes)))
                for image_file_name, bounding_box, is_fully_occluded, is_out_of_view in zip(image_file_names, bounding_boxes, is_fully_occluded_flags, is_out_of_view_flags):
                    with sequence_constructor.new_frame() as frame_constructor:
                        frame_constructor.set_path(os.path.join(image_folder_path, image_file_name), frame_size)
                        frame_constructor.set_bounding_box(bounding_box.tolist(), validity=not (is_fully_occluded or is_out_of_view))
                        frame_constructor.set_object_attribute('occlusion', is_fully_occluded.item())
                        frame_constructor.set_object_attribute('out of view', is_out_of_view.item())


class LaSOTExtensionAttributes:
    """
    A class to encapsulate LaSOT extension subset sequence attributes and their properties.
    """
    ATTRIBUTE_NAMES = (
        'Illumination Variation', 'Partial Occlusion', 'Deformation',
        'Motion Blur', 'Camera Motion', 'Rotation', 'Background Clutter',
        'Viewpoint Change', 'Scale Variation', 'Full Occlusion', 'Fast Motion',
        'Out-of-View', 'Low Resolution', 'Aspect Ration Change')
    ATTRIBUTE_SHORT_NAMES = ('IV', 'POC', 'DEF', 'MB', 'CM', 'ROT', 'BC', 'VC', 'SV', 'FOC', 'FM', 'OV', 'LR', 'ARC')

    SEQUENCE_ATTRIBUTES = {
        'atv-1': (False, False, False, False, False, True, False, True, True, False, False, False, False, False),
        'atv-2': (False, False, False, False, True, False, False, True, True, False, False, False, False, False),
        'atv-3': (True, False, False, True, True, False, False, True, True, False, False, False, False, False),
        'atv-4': (False, False, False, False, True, False, False, True, True, False, True, False, True, True),
        'atv-5': (False, True, False, False, False, True, False, True, True, True, False, False, False, True),
        'atv-6': (True, True, False, False, True, False, False, True, True, False, True, False, True, True),
        'atv-7': (False, True, False, False, True, False, False, True, True, True, True, True, False, True),
        'atv-8': (False, True, False, False, False, True, False, True, True, False, True, False, False, True),
        'atv-9': (False, True, False, False, False, False, False, True, True, False, False, False, False, True),
        'atv-10': (False, True, False, False, False, True, False, True, True, False, False, False, True, True),
        'badminton-1': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'badminton-2': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'badminton-3': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'badminton-4': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'badminton-5': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'badminton-6': (False, True, False, True, False, False, False, False, True, True, True, False, True, True),
        'badminton-7': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'badminton-8': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'badminton-9': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'badminton-10': (False, False, False, True, False, False, False, False, True, True, True, True, True, True),
        'cosplay-1': (False, True, True, True, False, True, False, False, True, False, False, False, False, True),
        'cosplay-2': (False, True, True, False, False, True, False, False, True, False, False, False, False, True),
        'cosplay-3': (False, True, True, False, False, True, False, False, True, True, True, False, True, True),
        'cosplay-4': (True, True, True, False, False, True, False, False, True, False, False, False, False, False),
        'cosplay-5': (False, True, True, False, False, True, False, False, True, False, False, False, False, False),
        'cosplay-6': (False, False, True, False, False, True, False, False, False, False, False, False, False, True),
        'cosplay-7': (False, True, True, False, False, True, False, False, True, False, False, False, True, True),
        'cosplay-8': (False, True, True, False, False, True, True, False, True, False, True, False, True, True),
        'cosplay-9': (True, False, True, False, False, True, False, False, True, False, False, False, False, False),
        'cosplay-10': (False, True, False, False, False, True, True, False, True, True, True, False, True, True),
        'dancingshoe-1': (False, True, False, True, False, True, True, False, True, True, True, False, True, True),
        'dancingshoe-2': (False, True, False, True, False, True, True, False, True, True, True, False, True, True),
        'dancingshoe-3': (False, True, False, True, False, True, True, True, True, True, True, False, True, True),
        'dancingshoe-4': (False, True, False, True, False, True, True, False, True, True, True, False, True, True),
        'dancingshoe-5': (False, True, False, False, False, True, True, True, True, True, True, False, True, True),
        'dancingshoe-6': (False, True, False, True, False, False, True, False, True, True, True, False, True, True),
        'dancingshoe-7': (False, True, False, True, False, True, True, True, True, True, True, False, True, True),
        'dancingshoe-8': (False, True, False, True, False, True, True, True, True, True, True, False, True, True),
        'dancingshoe-9': (False, True, False, True, False, True, True, True, True, True, True, False, True, True),
        'dancingshoe-10': (False, True, False, True, False, True, True, True, True, False, True, False, True, True),
        'footbag-1': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'footbag-2': (False, False, False, True, False, False, False, False, True, True, True, False, True, False),
        'footbag-3': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'footbag-4': (False, False, False, True, False, False, True, False, True, True, True, False, True, True),
        'footbag-5': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'footbag-6': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'footbag-7': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'footbag-8': (False, False, False, True, False, False, True, False, True, True, True, True, True, True),
        'footbag-9': (False, False, False, True, False, False, True, False, True, True, True, False, True, True),
        'footbag-10': (False, False, False, True, False, False, False, False, True, True, True, False, True, True),
        'frisbee-1': (False, True, False, True, False, False, False, True, True, True, True, False, True, True),
        'frisbee-2': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'frisbee-3': (False, True, False, True, False, False, False, True, True, True, False, False, True, True),
        'frisbee-4': (False, True, False, True, False, False, False, True, True, True, True, False, True, True),
        'frisbee-5': (False, True, False, True, False, False, False, True, True, True, True, False, True, True),
        'frisbee-6': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'frisbee-7': (False, True, False, False, False, False, False, True, True, True, True, False, True, True),
        'frisbee-8': (True, True, False, False, False, False, False, True, True, True, True, True, True, True),
        'frisbee-9': (False, True, False, False, False, False, True, True, True, True, True, True, True, True),
        'frisbee-10': (False, True, False, True, False, False, False, True, True, True, True, True, True, True),
        'jianzi-1': (False, True, True, False, False, False, False, False, True, True, True, False, True, True),
        'jianzi-2': (False, True, True, False, False, False, False, False, True, True, True, True, True, True),
        'jianzi-3': (True, True, True, True, False, False, True, False, True, True, True, True, True, True),
        'jianzi-4': (False, False, False, True, False, False, True, False, False, False, False, False, False, True),
        'jianzi-5': (False, True, False, True, True, False, True, False, True, True, True, True, True, True),
        'jianzi-6': (False, False, False, True, False, True, False, False, True, False, False, False, False, True),
        'jianzi-7': (False, True, False, True, False, False, False, False, True, True, True, True, True, True),
        'jianzi-8': (False, True, True, False, False, False, False, False, True, True, True, True, True, True),
        'jianzi-9': (False, True, False, False, False, True, False, False, True, True, True, True, True, True),
        'jianzi-10': (False, True, True, False, False, False, True, False, True, True, True, True, True, True),
        'lantern-1': (False, True, False, True, True, False, True, False, True, True, True, True, True, True),
        'lantern-2': (False, False, False, False, False, False, True, False, True, False, False, False, True, True),
        'lantern-3': (False, True, False, False, False, False, True, False, True, True, False, False, True, True),
        'lantern-4': (False, True, False, False, False, False, True, False, True, False, False, False, True, False),
        'lantern-5': (False, False, False, False, False, False, True, False, True, False, False, False, True, True),
        'lantern-6': (True, True, False, False, False, False, True, False, False, True, False, False, True, False),
        'lantern-7': (True, True, False, False, False, False, True, False, True, True, False, False, False, True),
        'lantern-8': (False, False, False, False, False, False, False, False, True, False, False, False, True, False),
        'lantern-9': (True, False, False, False, False, False, False, False, True, False, False, False, True, False),
        'lantern-10': (True, False, False, False, False, False, False, False, True, False, False, False, True, True),
        'misc-1': (False, True, False, True, False, True, True, False, True, True, True, False, True, False),
        'misc-2': (False, True, False, False, False, False, True, True, True, True, True, True, True, True),
        'misc-3': (False, True, True, True, False, False, False, False, True, True, True, True, True, True),
        'misc-4': (False, True, False, False, False, False, True, False, False, True, True, False, True, False),
        'misc-5': (False, False, True, False, False, False, False, True, True, False, False, False, False, True),
        'misc-6': (True, True, False, False, False, False, True, False, True, False, False, True, False, True),
        'misc-7': (False, False, False, False, False, True, False, True, True, False, True, False, True, True),
        'misc-8': (False, True, False, True, False, True, False, False, True, True, True, True, True, True),
        'misc-9': (False, False, False, False, False, True, False, False, True, True, True, True, True, True),
        'misc-10': (False, True, False, False, False, True, True, False, True, False, False, False, False, True),
        'opossum-1': (False, True, True, True, False, False, False, False, True, True, True, True, False, True),
        'opossum-2': (False, True, True, True, False, False, False, False, True, False, False, False, True, True),
        'opossum-3': (False, True, True, True, False, False, True, False, True, False, False, False, False, True),
        'opossum-4': (False, True, True, False, False, False, False, False, True, False, False, False, False, True),
        'opossum-5': (False, True, True, False, False, False, False, False, True, False, False, False, False, True),
        'opossum-6': (False, True, True, True, False, False, True, False, True, True, False, False, False, True),
        'opossum-7': (True, False, True, False, False, False, False, False, True, False, False, False, False, True),
        'opossum-8': (False, True, True, False, False, False, True, False, True, False, False, False, False, False),
        'opossum-9': (False, True, True, False, False, False, False, False, True, True, False, False, False, True),
        'opossum-10': (False, True, True, False, False, False, False, False, False, False, False, False, False, False),
        'paddle-1': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'paddle-2': (False, True, False, True, False, False, False, True, True, True, True, False, True, True),
        'paddle-3': (False, True, False, True, False, False, False, True, True, True, False, True, False, True),
        'paddle-4': (False, True, False, True, False, False, False, True, True, True, True, False, True, True),
        'paddle-5': (False, True, False, False, False, False, True, True, True, True, True, False, True, True),
        'paddle-6': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'paddle-7': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'paddle-8': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'paddle-9': (False, True, False, True, False, False, False, True, True, True, True, True, True, True),
        'paddle-10': (False, True, False, True, False, False, True, True, True, True, True, False, True, True),
        'raccoon-1': (False, True, True, True, False, False, False, True, True, False, False, False, False, True),
        'raccoon-2': (False, True, True, False, False, False, True, True, True, True, False, False, False, True),
        'raccoon-3': (False, True, True, False, False, False, True, True, True, False, False, False, False, True),
        'raccoon-4': (True, True, True, False, False, False, False, True, True, True, False, False, False, True),
        'raccoon-5': (False, True, True, False, False, False, True, True, True, True, True, False, True, True),
        'raccoon-6': (False, False, True, False, False, False, False, True, True, False, False, False, False, True),
        'raccoon-7': (False, False, True, False, False, False, True, True, False, False, False, False, False, True),
        'raccoon-8': (False, True, True, False, False, False, True, True, True, False, False, False, False, True),
        'raccoon-9': (False, False, True, False, True, False, False, False, False, False, False, False, False, False),
        'raccoon-10': (False, True, True, False, False, False, True, True, True, True, False, False, True, True),
        'rhino-1': (False, True, True, False, False, False, False, False, True, False, True, False, True, True),
        'rhino-2': (False, False, True, False, False, False, False, True, True, False, False, False, True, True),
        'rhino-3': (False, False, False, False, False, False, True, False, False, False, False, False, False, False),
        'rhino-4': (False, True, False, False, False, False, True, False, True, False, False, False, False, False),
        'rhino-5': (False, False, True, False, True, False, False, False, True, False, True, False, True, True),
        'rhino-6': (False, False, True, False, False, False, True, False, False, False, False, False, False, False),
        'rhino-7': (False, False, False, False, False, True, False, False, True, False, False, False, False, True),
        'rhino-8': (True, True, True, False, False, False, False, False, True, True, True, False, True, True),
        'rhino-9': (False, False, True, False, False, False, False, False, True, False, False, False, False, True),
        'rhino-10': (False, True, True, False, False, False, False, True, True, False, False, True, False, True),
        'skatingshoe-1': (False, True, False, True, False, False, True, True, True, False, False, False, True, False),
        'skatingshoe-2': (False, True, False, True, False, True, True, False, True, True, True, False, True, True),
        'skatingshoe-3': (False, True, False, True, False, True, True, False, True, True, False, True, False, True),
        'skatingshoe-4': (False, False, False, False, False, True, True, False, True, False, False, False, False, True),
        'skatingshoe-5': (False, True, False, False, False, False, True, False, True, True, True, False, True, True),
        'skatingshoe-6': (False, True, False, False, False, False, True, False, True, False, False, False, True, True),
        'skatingshoe-7': (False, True, False, True, False, True, True, False, True, True, True, True, True, True),
        'skatingshoe-8': (False, True, False, True, False, True, True, False, True, True, True, True, True, True),
        'skatingshoe-9': (False, True, False, False, False, True, True, False, True, True, False, False, True, True),
        'skatingshoe-10': (False, True, False, True, False, True, True, False, True, True, True, False, True, True),
        'wingsuit-1': (False, False, False, False, True, False, False, True, True, False, True, False, True, True),
        'wingsuit-2': (False, True, False, False, False, False, True, False, True, True, False, False, True, True),
        'wingsuit-3': (False, False, False, False, True, False, True, True, True, True, True, False, True, False),
        'wingsuit-4': (False, False, False, False, True, False, True, True, True, True, True, False, True, True),
        'wingsuit-5': (False, True, False, False, True, False, False, False, True, True, True, False, True, True),
        'wingsuit-6': (False, False, False, False, True, False, True, True, True, True, True, False, True, True),
        'wingsuit-7': (False, False, False, False, True, True, False, True, True, True, True, False, True, True),
        'wingsuit-8': (False, True, False, False, True, False, False, True, True, False, False, False, False, True),
        'wingsuit-9': (False, True, False, False, True, False, True, True, True, True, True, False, True, True),
        'wingsuit-10': (False, False, False, False, True, False, True, True, True, True, False, False, True, True)
    }

_categories = ["atv", "badminton", "cosplay", "dancingshoe", "footbag", "frisbee", "jianzi", "lantern", "misc", "opossum", "paddle", "raccoon", "rhino", "skatingshoe", "wingsuit"]
_sequence_frame_size = {'atv-1': [1280, 720], 'atv-2': [1280, 720], 'atv-3': [1280, 720], 'atv-4': [1280, 720], 'atv-5': [1280, 720], 'atv-6': [1280, 720], 'atv-7': [1280, 720], 'atv-8': [1280, 720], 'atv-9': [1280, 720], 'atv-10': [1280, 720], 'badminton-1': [1280, 720], 'badminton-2': [1280, 720], 'badminton-3': [1280, 720], 'badminton-4': [1280, 720], 'badminton-5': [1280, 720], 'badminton-6': [1280, 720], 'badminton-7': [1280, 720], 'badminton-8': [1280, 720], 'badminton-9': [1280, 720], 'badminton-10': [1280, 720], 'cosplay-1': [640, 480], 'cosplay-2': [640, 480], 'cosplay-3': [640, 480], 'cosplay-4': [854, 480], 'cosplay-5': [854, 480], 'cosplay-6': [854, 480], 'cosplay-7': [640, 480], 'cosplay-8': [640, 480], 'cosplay-9': [1280, 720], 'cosplay-10': [1280, 720], 'dancingshoe-1': [1280, 720], 'dancingshoe-2': [1280, 720], 'dancingshoe-3': [1280, 720], 'dancingshoe-4': [1280, 720], 'dancingshoe-5': [1280, 720], 'dancingshoe-6': [1280, 720], 'dancingshoe-7': [1280, 720], 'dancingshoe-8': [1280, 720], 'dancingshoe-9': [1280, 720], 'dancingshoe-10': [1280, 720], 'footbag-1': [1280, 720], 'footbag-2': [1280, 720], 'footbag-3': [1271, 720], 'footbag-4': [1280, 720], 'footbag-5': [1280, 720], 'footbag-6': [1280, 720], 'footbag-7': [1280, 720], 'footbag-8': [960, 720], 'footbag-9': [1280, 720], 'footbag-10': [1280, 720], 'frisbee-1': [1280, 720], 'frisbee-2': [1280, 720], 'frisbee-3': [1280, 720], 'frisbee-4': [1280, 720], 'frisbee-5': [1280, 720], 'frisbee-6': [1280, 720], 'frisbee-7': [1280, 720], 'frisbee-8': [1280, 720], 'frisbee-9': [1280, 720], 'frisbee-10': [1280, 720], 'jianzi-1': [1920, 1080], 'jianzi-2': [1280, 720], 'jianzi-3': [1920, 1080], 'jianzi-4': [1920, 1080], 'jianzi-5': [1280, 720], 'jianzi-6': [1280, 720], 'jianzi-7': [1280, 720], 'jianzi-8': [638, 360], 'jianzi-9': [276, 480], 'jianzi-10': [640, 480], 'lantern-1': [1280, 720], 'lantern-2': [1280, 720], 'lantern-3': [1280, 720], 'lantern-4': [1280, 720], 'lantern-5': [1280, 720], 'lantern-6': [1280, 720], 'lantern-7': [1280, 720], 'lantern-8': [1280, 720], 'lantern-9': [1280, 720], 'lantern-10': [1280, 720], 'misc-1': [1280, 720], 'misc-2': [1280, 720], 'misc-3': [1280, 720], 'misc-4': [1280, 720], 'misc-5': [1280, 720], 'misc-6': [1280, 720], 'misc-7': [1280, 720], 'misc-8': [1280, 720], 'misc-9': [1280, 720], 'misc-10': [1280, 720], 'opossum-1': [1280, 720], 'opossum-2': [304, 540], 'opossum-3': [1280, 720], 'opossum-4': [1280, 720], 'opossum-5': [1280, 720], 'opossum-6': [720, 480], 'opossum-7': [720, 480], 'opossum-8': [854, 480], 'opossum-9': [1280, 720], 'opossum-10': [1280, 720], 'paddle-1': [1280, 720], 'paddle-2': [1280, 720], 'paddle-3': [1280, 720], 'paddle-4': [1280, 720], 'paddle-5': [1280, 720], 'paddle-6': [1280, 720], 'paddle-7': [1280, 720], 'paddle-8': [1280, 720], 'paddle-9': [1280, 720], 'paddle-10': [1280, 720], 'raccoon-1': [1280, 720], 'raccoon-2': [1280, 720], 'raccoon-3': [640, 480], 'raccoon-4': [1280, 720], 'raccoon-5': [640, 360], 'raccoon-6': [1280, 720], 'raccoon-7': [1280, 720], 'raccoon-8': [1280, 720], 'raccoon-9': [1280, 720], 'raccoon-10': [640, 480], 'rhino-1': [854, 480], 'rhino-2': [1280, 720], 'rhino-3': [1280, 720], 'rhino-4': [1280, 720], 'rhino-5': [1280, 720], 'rhino-6': [1280, 720], 'rhino-7': [1280, 720], 'rhino-8': [1280, 720], 'rhino-9': [1280, 720], 'rhino-10': [1280, 720], 'skatingshoe-1': [1920, 1080], 'skatingshoe-2': [1920, 1080], 'skatingshoe-3': [1920, 1080], 'skatingshoe-4': [1920, 1080], 'skatingshoe-5': [1920, 1080], 'skatingshoe-6': [1920, 1080], 'skatingshoe-7': [1920, 1080], 'skatingshoe-8': [1920, 1080], 'skatingshoe-9': [1280, 720], 'skatingshoe-10': [1920, 1080], 'wingsuit-1': [1280, 720], 'wingsuit-2': [1920, 1080], 'wingsuit-3': [1280, 720], 'wingsuit-4': [960, 720], 'wingsuit-5': [1280, 720], 'wingsuit-6': [1280, 720], 'wingsuit-7': [1280, 720], 'wingsuit-8': [1280, 720], 'wingsuit-9': [1280, 720], 'wingsuit-10': [1280, 720]}
