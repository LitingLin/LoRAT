from __future__ import annotations
from trackit.datasets.MOT.specialization.memory_mapped.dataset import MultipleObjectTrackingDataset_MemoryMapped,\
    MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetTrack_MemoryMapped,\
    MultipleObjectTrackingDatasetFrame_MemoryMapped, MultipleObjectTrackingDatasetFrameObject_MemoryMapped
from trackit.data.source import *
from trackit.miscellanies.image.io import *


class MOTObject(TrackingDataset_Object):
    def __init__(self, object_: MultipleObjectTrackingDatasetFrameObject_MemoryMapped):
        self._object = object_

    def get_object_id(self) -> int:
        return self._object.get_id()

    def get_category_id(self) -> Optional[int]:
        return self._object.get_category_id() if self._object.has_category_id() else None

    def get_bounding_box(self) -> np.ndarray:
        if not self._object.has_bounding_box():
            return np.full((4,), float('nan'), np.float64)
        return self._object.get_bounding_box().astype(np.float64)

    def get_existence_flag(self) -> bool:
        if not self._object.has_bounding_box():
            return False
        return self._object.get_bounding_box_validity_flag().item() if self._object.has_bounding_box_validity_flag() else True

    def get_object_attribute(self, name: str) -> Any:
        return self._object.get_attribute(name)

    def get_all_object_attribute_name(self) -> Tuple[str, ...]:
        return self._object.get_all_attribute_name()


class MOTFrameInTrack(TrackingDataset_FrameInTrack):
    def __init__(self, frame: MultipleObjectTrackingDatasetFrame_MemoryMapped, id_: int, frame_index: int,
                 object_: Optional[MultipleObjectTrackingDatasetFrameObject_MemoryMapped],
                 ):
        self._frame = frame
        self._id = id_
        self._frame_index = frame_index
        self._object = object_

    def get_object_id(self) -> int:
        return self._id

    def get_category_id(self) -> Optional[int]:
        if self._object is not None and self._object.has_category_id():
            return self._object.get_category_id()
        else:
            return None

    def get_frame_size(self) -> np.ndarray:
        return self._frame.get_image_size()

    def get_frame(self) -> np.ndarray:
        return read_image_with_auto_retry(self._frame.get_image_path())

    def get_frame_file_path(self) -> str:
        return self._frame.get_image_path()

    def get_bounding_box(self) -> np.ndarray:
        if self._object is None or not self._object.has_bounding_box():
            return np.full((4,), float('nan'), np.float64)

        return self._object.get_bounding_box().astype(np.float64)

    def get_existence_flag(self) -> bool:
        if self._object is None or not self._object.has_bounding_box():
            return False
        if not self._object.has_bounding_box_validity_flag():
            return True
        return self._object.get_bounding_box_validity_flag().item()

    def get_frame_index(self) -> int:
        return self._frame_index

    def get_frame_index_in_sequence(self) -> int:
        return self._frame.get_frame_index()

    def is_file_backed(self) -> bool:
        return True

    def get_object_attribute(self, name: str) -> Any:
        if self._object is not None:
            return self._object.get_attribute(name)
        else:
            return None

    def get_all_object_attribute_name(self) -> Tuple[str, ...]:
        if self._object is not None:
            return self._object.get_all_attribute_name()
        else:
            return tuple()

    def get_frame_attribute(self, name: str) -> Any:
        return self._frame.get_attribute(name)

    def get_all_frame_attribute_name(self) -> Tuple[str, ...]:
        return self._frame.get_all_attribute_name()


class MOTFrame(TrackingDataset_Frame):
    def __init__(self, frame: MultipleObjectTrackingDatasetFrame_MemoryMapped):
        self._frame = frame

    def is_file_backed(self) -> bool:
        return True

    def get_number_of_objects(self) -> int:
        return len(self._frame)

    def get_all_object_id(self) -> Tuple[int, ...]:
        return self._frame.get_all_object_ids().tolist()

    def get_object_by_index(self, index: int) -> TrackingDataset_Object:
        object_ = self._frame[index]
        return MOTObject(object_)

    def get_object_by_id(self, id_) -> TrackingDataset_Object:
        object_ = self._frame.get_object_by_id(id_)
        return MOTObject(object_)

    def get_frame_size(self) -> np.ndarray:
        return self._frame.get_image_size()

    def get_frame(self) -> np.ndarray:
        return read_image_with_auto_retry(self._frame.get_image_path())

    def get_frame_file_path(self) -> str:
        return self._frame.get_image_path()

    def get_frame_index(self) -> int:
        return self._frame.get_frame_index()

    def get_frame_attribute(self, name: str) -> Any:
        return self._frame.get_attribute(name)

    def get_all_frame_attribute_name(self) -> Tuple[str, ...]:
        return self._frame.get_all_attribute_name()


class MOTObjectTrack(TrackingDataset_Track):
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, track: MultipleObjectTrackingDatasetTrack_MemoryMapped):
        self._sequence = sequence
        self._track = track

    def get_name(self) -> str:
        name = self._sequence.get_name()
        return name + '-' + str(self._track.get_id())

    def get_all_frame_size(self) -> np.ndarray:
        frame_indices = self._track.get_all_frame_indices()
        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]
        return self._sequence.get_all_frame_sizes()[first_frame: last_frame + 1]

    def get_all_object_bounding_box(self) -> np.ndarray:
        frame_indices = self._track.get_all_frame_indices()
        length = len(self)
        frame_indices = frame_indices - frame_indices[0]
        all_bounding_box = np.full(length, float('nan'), dtype=np.float64)
        all_annotated_bounding_box = self._track.get_all_bounding_boxes()
        if all_annotated_bounding_box is not None:
            all_bounding_box[frame_indices] = all_annotated_bounding_box
        return all_bounding_box

    def get_all_object_existence_flag(self) -> Optional[np.ndarray]:
        frame_indices = self._track.get_all_frame_indices()
        length = len(self)
        frame_indices = frame_indices - frame_indices[0]
        all_existence_flag = np.zeros(length, dtype=np.bool_)

        all_visible_existence_flag = self._track.get_all_bounding_box_validity_flags()
        if all_visible_existence_flag is None:
            all_existence_flag[frame_indices] = True
        else:
            all_existence_flag[frame_indices] = all_visible_existence_flag
        return all_existence_flag

    def get_object_id(self) -> int:
        return self._track.get_id()

    def get_category_id(self) -> Optional[int]:
        return self._track.get_category_id() if self._track.has_category_id() else None

    def __len__(self) -> int:
        frame_indices = self._track.get_all_frame_indices()
        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]
        return last_frame - first_frame + 1

    def __getitem__(self, index: int) -> TrackingDataset_FrameInTrack:
        frame_indices = self._track.get_all_frame_indices()
        first_frame = frame_indices[0]
        frame = self._sequence[index + first_frame]
        if frame.has_object(self._track.get_id()):
            object_ = frame.get_object_by_id(self._track.get_id())
        else:
            object_ = None
        return MOTFrameInTrack(frame, self._track.get_id(), index, object_)

    def get_attribute(self, name: str) -> Any:
        return self._track.get_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._track.get_all_attribute_name()


class MOTSequence(TrackingDataset_Sequence):
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped):
        self._sequence = sequence

    def get_name(self) -> str:
        return self._sequence.get_name()

    def get_fps(self) -> Optional[float]:
        if not self._sequence.has_fps():
            return None
        return self._sequence.get_fps()

    def get_number_of_tracks(self) -> int:
        return self._sequence.get_number_of_objects()

    def __getitem__(self, index: int) -> MOTFrame:
        return MOTFrame(self._sequence[index])

    def __len__(self) -> int:
        return len(self._sequence)

    def get_track_by_index(self, index: int) -> TrackingDataset_Track:
        return MOTObjectTrack(self._sequence, self._sequence.get_object(index))

    def get_track_by_id(self, id_: int) -> TrackingDataset_Track:
        return MOTObjectTrack(self._sequence, self._sequence.get_object_by_id(id_))

    def get_all_object_id(self) -> Tuple[int, ...]:
        return self._sequence.get_all_object_ids()

    def get_attribute(self, name: str) -> Any:
        return self._sequence.get_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._sequence.get_all_attribute_name()


class MOTDataset(TrackingDataset):
    def __init__(self, dataset: MultipleObjectTrackingDataset_MemoryMapped):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> TrackingDataset_Sequence:
        return MOTSequence(self._dataset[index])

    def get_category_name_by_id(self, id_: int) -> Optional[str]:
        if self._dataset.has_category_id_name_map():
            return self._dataset.get_category_name_by_id(id_)
        else:
            return None

    def get_all_category_id_name_map(self) -> Optional[Mapping[int, str]]:
        if self._dataset.has_category_id_name_map():
            return self._dataset.get_category_id_name_map()
        else:
            return None

    def get_name(self) -> str:
        return self._dataset.get_name()

    def get_data_split(self) -> Tuple[str, ...]:
        return self._dataset.get_data_split()

    def get_full_name(self) -> str:
        return self._dataset.get_full_name()

    def get_attribute(self, name: str) -> Any:
        return self._dataset.get_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._dataset.get_all_attribute_name()
