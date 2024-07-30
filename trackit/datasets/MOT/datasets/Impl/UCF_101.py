import os
import scipy.io
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor


def construct_UCF101THUMOSDataset(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    annotation_file_path = seed.annotation_file_path

    annotation = scipy.io.loadmat(annotation_file_path)

    annotation = annotation['annot']
    assert annotation.shape[0] == 1
    constructor.set_total_number_of_sequences(annotation.shape[1])
    constructor.set_category_id_name_map({0: 'person'})
    for index_of_sequence in range(annotation.shape[1]):
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            sequence_annotation = annotation[0][index_of_sequence]
            sequence_name = sequence_annotation['name'][0].item()
            assert type(sequence_name) == str
            number_of_images = sequence_annotation['num_imgs'][0][0].item()
            assert type(number_of_images) == int

            images_path = os.path.join(root_path, sequence_name)
            images = os.listdir(images_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            assert len(images) == number_of_images
            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(images_path, image))

            assert sequence_annotation['tubes'].shape[0] == 1
            number_of_objects = sequence_annotation['tubes'].shape[1]
            for index_of_object in range(number_of_objects):
                with sequence_constructor.new_object(index_of_object) as object_constructor:
                    object_constructor.set_category_id(0)

                object_annotation = sequence_annotation['tubes'][0][index_of_object]

                max_frame = object_annotation['ef'][0][0].item()
                min_frame = object_annotation['sf'][0][0].item() - 1
                number_of_frames = max_frame - min_frame
                assert number_of_frames == object_annotation['boxes'].shape[0]
                for index_of_frame in range(number_of_frames):
                    bounding_box = object_annotation['boxes'][index_of_frame].tolist()
                    truly_index_of_frame = index_of_frame + min_frame
                    with sequence_constructor.open_frame(truly_index_of_frame) as frame_constructor:
                        with frame_constructor.new_object(index_of_object) as object_constructor:
                            object_constructor.set_bounding_box(bounding_box)
