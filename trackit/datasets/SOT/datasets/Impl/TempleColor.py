import os
import numpy as np
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


def construct_TempleColor(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    sequence_label_mapper = {
        'Airport_ce': 'person',
        'Baby_ce': 'baby',
        'Badminton_ce1': 'person',
        'Badminton_ce2': 'person',
        'Ball_ce1': 'ball',
        'Ball_ce2': 'ball',
        'Ball_ce3': 'ball',
        'Ball_ce4': 'ball',
        'Basketball': 'person',
        'Basketball_ce1': 'person',
        'Basketball_ce2': 'person',
        'Basketball_ce3': 'person',
        'Bee_ce': '8bit',
        'Bicycle': 'person',
        'Bike_ce1': 'person',
        'Bike_ce2': 'person',
        'Biker': 'person',
        'Bikeshow_ce': 'person',
        'Bird': 'bird',
        'Board': 'circuit board',
        'Boat_ce1': 'boat',
        'Boat_ce2': 'boat',
        'Bolt': 'person',
        'Boy': 'head',
        'Busstation_ce1': 'person',
        'Busstation_ce2': 'person',
        'Carchasing_ce1': 'car',
        'Carchasing_ce3': 'car',
        'Carchasing_ce4': 'car',
        'CarDark': 'car',
        'CarScale': 'car',
        'Charger_ce': 'charger',
        'Coke': 'can',
        'Couple': 'person group',
        'Crossing': 'person',
        'Cup': 'cup',
        'Cup_ce': 'cup',
        'David': 'face',
        'David3': 'person',
        'Deer': 'dear head',
        'Diving': 'person',
        'Doll': 'doll',
        'Eagle_ce': 'eagle',
        'Electricalbike_ce': 'motorcycle',
        'Face_ce': 'head',
        'Face_ce2': 'face',
        'FaceOcc1': 'face',
        'Fish_ce1': 'fish',
        'Fish_ce2': 'fish',
        'Football1': 'face',
        'Girl': 'head',
        'Girlmov': 'child',
        'Guitar_ce1': 'guitar',
        'Guitar_ce2': 'guitar',
        'Gym': 'body',
        'Hand': 'hand',
        'Hand_ce1': 'hand',
        'Hand_ce2': 'hand',
        'Hurdle_ce1': 'body',
        'Hurdle_ce2': 'body',
        'Iceskater': 'person',
        'Ironman': 'head',
        'Jogging1': 'person',
        'Jogging2': 'person',
        'Juice': 'box',
        'Kite_ce1': 'kite',
        'Kite_ce2': 'kite',
        'Kite_ce3': 'kite',
        'Kobe_ce': 'person',
        'Lemming': 'toy',
        'Liquor': 'bottle',
        'Logo_ce': 'logo',
        'Matrix': 'head',
        'Messi_ce': 'person',
        'Michaeljackson_ce': 'body',
        'Microphone_ce1': 'microphone',
        'Microphone_ce2': 'microphone',
        'MotorRolling': 'motorcycle',
        'Motorbike_ce': 'person',
        'MountainBike': 'bicycle',
        'Panda': 'head',
        'Plane_ce2': 'airplane',
        'Plate_ce1': 'license plate',
        'Plate_ce2': 'license plate',
        'Pool_ce1': 'pool',
        'Pool_ce2': 'pool',
        'Pool_ce3': 'pool',
        'Railwaystation_ce': 'person',
        'Ring_ce': 'ring',
        'Sailor_ce': '8bit',
        'Shaking': 'head',
        'Singer_ce1': 'person',
        'Singer_ce2': 'body',
        'Singer1': 'person',
        'Singer2': 'body',
        'Skating1': 'person',
        'Skating2': 'person',
        'Skating_ce1': 'body',
        'Skating_ce2': 'body',
        'Skiing': 'person',
        'Skiing_ce': 'person',
        'Skyjumping_ce': 'person',
        'Soccer': 'head',
        'Spiderman_ce': 'body',
        'Subway': 'person',
        'Suitcase_ce': 'suitcase',
        'Sunshade': 'head',
        'SuperMario_ce': '8bit',
        'Surf_ce1': 'body',
        'Surf_ce2': 'body',
        'Surf_ce3': 'body',
        'Surf_ce4': 'body',
        'TableTennis_ce': 'table tennis ball',
        'Tennis_ce1': 'body',
        'Tennis_ce2': 'body',
        'Tennis_ce3': 'body',
        'TennisBall_ce': 'tennis ball',
        'Thunder_ce': '8bit',
        'Tiger1': 'toy',
        'Tiger2': 'toy',
        'Torus': 'torus',
        'Toyplane_ce': 'airplane',
        'Trellis': 'head',
        'Walking': 'person',
        'Walking2': 'person',
        'Woman': 'person',
        'Yo-yos_ce1': 'yo-yo',
        'Yo-yos_ce2': 'yo-yo',
        'Yo-yos_ce3': 'yo-yo'
    }

    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list.sort()

    number_of_sequences = len(sequence_list)
    category_names = set(sequence_label_mapper.values())
    category_id_name_map = {i: v for i, v in enumerate(category_names)}
    category_name_id_map = {v: i for i, v in enumerate(category_names)}
    constructor.set_category_id_name_map(category_id_name_map)
    constructor.set_total_number_of_sequences(number_of_sequences)

    for index in range(number_of_sequences):
        sequence_name = sequence_list[index]

        path = os.path.join(root_path, sequence_name)
        img_path = os.path.join(path, 'img')

        ground_truth_file = os.path.join(path, '{}_gt.txt'.format(sequence_name))
        try:
            frames_file = os.path.join(path, '{}_frames.txt'.format(sequence_name))
            with open(frames_file) as fid:
                frames_file_content = fid.read()
        except FileNotFoundError:
            frames_file = os.path.join(path, '{}_frames.txt'.format(sequence_name.lower()))
            with open(frames_file) as fid:
                frames_file_content = fid.read()
        frames_file_content = frames_file_content.strip()
        frame_indices = frames_file_content.split(',')
        assert len(frame_indices) == 2
        start_index = int(frame_indices[0])
        end_index = int(frame_indices[1]) + 1

        bounding_boxes = np.loadtxt(os.path.join(path, ground_truth_file), dtype=np.float64, delimiter=',')
        assert end_index - start_index == len(bounding_boxes)

        images = os.listdir(img_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        with constructor.new_sequence(category_name_id_map[sequence_label_mapper[sequence_name]]) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(img_path, image))

            for index, index_of_frame in enumerate(range(start_index, end_index)):
                image_file_name = '{:04}.jpg'.format(index_of_frame)
                index_of_image = images.index(image_file_name)
                with sequence_constructor.open_frame(index_of_image) as frame_constructor:
                    frame_constructor.set_bounding_box(bounding_boxes[index].tolist())
