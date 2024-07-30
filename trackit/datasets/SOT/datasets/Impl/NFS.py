import os
from trackit.datasets.SOT.datasets.NFS import NFSDatasetVersionFlag
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def construct_NFS(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    version = seed.nfs_version
    manual_anno_only = seed.manual_anno_only

    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list = [dirname for dirname in sequence_list if
                 os.path.exists(os.path.join(root_path, dirname, '30', dirname)) and os.path.exists(
                     os.path.join(root_path, dirname, '240', dirname))]

    if version == NFSDatasetVersionFlag.fps_30:
        subDirName = '30'
    elif version == NFSDatasetVersionFlag.fps_240:
        subDirName = '240'
    else:
        raise Exception

    constructor.set_total_number_of_sequences(len(sequence_list))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for sequence_name in sequence_list:
        sequence_images_path = os.path.join(root_path, sequence_name, subDirName, sequence_name)
        sequence_anno_file_path = os.path.join(root_path, sequence_name, subDirName, '{}.txt'.format(sequence_name))
        images = os.listdir(sequence_images_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        className = None
        track_id = None
        bounding_box_annotations = {}
        for line_count, line in enumerate(open(sequence_anno_file_path)):
            '''
            # https://github.com/cvondrick/vatic
            1   Track ID. All rows with the same ID belong to the same path.
            2   xmin. The top left x-coordinate of the bounding box.
            3   ymin. The top left y-coordinate of the bounding box.
            4   xmax. The bottom right x-coordinate of the bounding box.
            5   ymax. The bottom right y-coordinate of the bounding box.
            6   frame. The frame that this annotation represents.
            7   lost. If 1, the annotation is outside of the view screen.
            8   occluded. If 1, the annotation is occluded.
            9   generated. If 1, the annotation was automatically interpolated.
            10  label. The label for this annotation, enclosed in quotation marks.
            11+ attributes. Each column after this is an attribute.
            '''
            line = line.strip()
            first_quote_index = line.find('"')
            if first_quote_index == -1:
                raise Exception
            attributes = line[:first_quote_index].split()
            track_id_ = int(attributes[0])
            if track_id is None:
                track_id = track_id_
            else:
                assert track_id_ == track_id
            bbox = attributes[1:5]
            bbox = [int(v) for v in bbox]
            frame_index = int(attributes[5]) - 1
            if sequence_name == 'pingpong_2' and frame_index >= 263:
                continue
            if subDirName == '30':
                if frame_index % 8 != 0:
                    continue
                else:
                    frame_index /= 8
            out_of_view = bool(int(attributes[6]))
            occluded = bool(int(attributes[7]))
            generated = bool(int(attributes[8]))
            if generated and manual_anno_only:
                continue
            bounding_box_annotations[frame_index] = (bbox, out_of_view, occluded)
            second_quote_index = line.rfind('"')
            if second_quote_index == -1 or second_quote_index <= first_quote_index:
                raise Exception
            # class name not accurate, ignore
            current_class_name = line[first_quote_index + 1: second_quote_index]
            if className is None:
                className = current_class_name
            elif className != current_class_name:
                raise Exception
        if len(bounding_box_annotations) == 0:
            continue
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            sequence_frame_size = _sequence_frame_size[sequence_name]
            for index_of_image, image in enumerate(images):
                image_path = os.path.join(sequence_images_path, image)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path, sequence_frame_size)
                    if index_of_image in bounding_box_annotations:
                        bbox, out_of_view, occluded = bounding_box_annotations[index_of_image]
                        frame_constructor.set_bounding_box(bbox, validity=not (out_of_view or occluded))
                        frame_constructor.set_object_attribute('lost', out_of_view)
                        frame_constructor.set_object_attribute('occluded', occluded)


_sequence_frame_size = {'Gymnastics': [1280, 720], 'MachLoop_jet': [1280, 720], 'Skiing_red': [1280, 720], 'Skydiving': [1920, 1080], 'airboard_1': [1280, 720], 'airplane_landing': [1280, 720], 'airtable_3': [1280, 720], 'basketball_1': [1280, 720], 'basketball_2': [1280, 720], 'basketball_3': [1280, 720], 'basketball_6': [1280, 720], 'basketball_7': [1280, 720], 'basketball_player': [1280, 720], 'basketball_player_2': [1280, 720], 'beach_flipback_person': [1280, 720], 'bee': [1280, 720], 'biker_acrobat': [1280, 720], 'biker_all_1': [1280, 720], 'biker_head_2': [1280, 720], 'biker_head_3': [1280, 720], 'biker_upper_body': [720, 1280], 'biker_whole_body': [1280, 720], 'billiard_2': [1280, 720], 'billiard_3': [1280, 720], 'billiard_6': [1280, 720], 'billiard_7': [1280, 720], 'billiard_8': [1280, 720], 'bird_2': [1280, 720], 'book': [1280, 720], 'bottle': [720, 1280], 'bowling_1': [720, 1280], 'bowling_2': [720, 1280], 'bowling_3': [1280, 720], 'bowling_6': [1280, 720], 'bowling_ball': [720, 1280], 'bunny': [1280, 720], 'car': [1280, 720], 'car_camaro': [1280, 720], 'car_drifting': [1280, 720], 'car_jumping': [1280, 720], 'car_rc_rolling': [1280, 720], 'car_rc_rotating': [1280, 720], 'car_side': [1280, 720], 'car_white': [1280, 720], 'cheetah': [1280, 720], 'cup': [1280, 720], 'cup_2': [720, 1280], 'dog': [1280, 720], 'dog_1': [1280, 720], 'dog_2': [1280, 720], 'dog_3': [720, 1280], 'dogs': [1920, 1080], 'dollar': [1280, 720], 'drone': [1920, 1080], 'ducks_lake': [1280, 720], 'exit': [1280, 720], 'first': [1280, 720], 'flower': [1280, 720], 'footbal_skill': [1920, 1080], 'helicopter': [1280, 720], 'horse_jumping': [1280, 720], 'horse_running': [1280, 720], 'iceskating_6': [1280, 720], 'jellyfish_5': [1280, 720], 'kid_swing': [1280, 720], 'motorcross': [1280, 720], 'motorcross_kawasaki': [1280, 720], 'parkour': [1280, 720], 'person_scooter': [1280, 720], 'pingpong_2': [1280, 720], 'pingpong_7': [1280, 720], 'pingpong_8': [1280, 720], 'purse': [720, 1280], 'rubber': [720, 1280], 'running': [1280, 720], 'running_100_m': [2048, 1080], 'running_100_m_2': [2048, 1080], 'running_2': [1280, 720], 'shuffleboard_1': [1280, 720], 'shuffleboard_2': [1280, 720], 'shuffleboard_4': [1280, 720], 'shuffleboard_5': [1280, 720], 'shuffleboard_6': [720, 1280], 'shuffletable_2': [720, 1280], 'shuffletable_3': [720, 1280], 'shuffletable_4': [720, 1280], 'ski_long': [1280, 720], 'soccer_ball': [720, 1280], 'soccer_ball_2': [720, 1280], 'soccer_ball_3': [720, 1280], 'soccer_player_2': [1280, 720], 'soccer_player_3': [1280, 720], 'stop_sign': [720, 1280], 'suv': [1280, 720], 'tiger': [1280, 720], 'walking': [720, 1280], 'walking_3': [720, 1280], 'water_ski_2': [1280, 720], 'yoyo': [720, 1280], 'zebra_fish': [1280, 720]}
