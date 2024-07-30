from __future__ import annotations
from trackit.datasets.SOT.specialization.memory_mapped.dataset import SingleObjectTrackingDataset_MemoryMapped, SingleObjectTrackingDatasetSequence_MemoryMapped, SingleObjectTrackingDatasetFrame_MemoryMapped
from trackit.data.source import *
from trackit.miscellanies.image.io import *


class SOTFrame(TrackingDataset_FrameInTrack, TrackingDataset_Frame):
    def __init__(self, frame: SingleObjectTrackingDatasetFrame_MemoryMapped, category_id: Optional[int]):
        self._frame = frame
        self._category_id = category_id

    def is_file_backed(self) -> bool:
        return True

    def get_frame_index_in_sequence(self) -> int:
        return self._frame.get_frame_index()

    def get_frame_index(self) -> int:
        return self._frame.get_frame_index()

    def get_object_id(self) -> int:
        return 0

    def get_number_of_objects(self) -> int:
        return 1

    def get_all_object_id(self) -> Tuple[int, ...]:
        return 0,

    def get_object_by_index(self, index: int) -> TrackingDataset_Object:
        assert index == 0
        return self

    def get_object_by_id(self, id_: int) -> TrackingDataset_Object:
        assert id_ == 0
        return self

    def get_frame_size(self) -> np.ndarray:
        return self._frame.get_image_size()

    def get_frame(self) -> np.ndarray:
        return read_image_with_auto_retry(self._frame.get_image_path())

    def get_frame_file_path(self) -> str:
        return self._frame.get_image_path()

    def get_category_id(self) -> Optional[int]:
        return self._category_id

    def get_bounding_box(self) -> np.ndarray:
        if not self._frame.has_bounding_box():
            return np.full((4, ), float('nan'), dtype=np.float64)
        return self._frame.get_bounding_box().astype(np.float64)

    def get_existence_flag(self) -> bool:
        if not self._frame.has_bounding_box():
            return False
        return self._frame.get_bounding_box_validity_flag().item() if self._frame.has_bounding_box_validity_flag() else True

    def get_object_attribute(self, name: str) -> Any:
        return self.get_object_attribute(name)

    def get_all_object_attribute_name(self) -> Tuple[str, ...]:
        return self.get_all_object_attribute_name()

    def get_frame_attribute(self, name: str) -> Any:
        return self.get_frame_attribute(name)

    def get_all_frame_attribute_name(self) -> Tuple[str, ...]:
        return self.get_all_frame_attribute_name()


class SOTTrack(TrackingDataset_Track):
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, category_id: Optional[int]):
        self._sequence = sequence
        self._category_id = category_id

    def get_object_id(self) -> int:
        return 0

    def get_category_id(self) -> Optional[int]:
        return self._category_id

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, index: int) -> TrackingDataset_FrameInTrack:
        return SOTFrame(self._sequence[index], self._category_id)

    def get_all_frame_size(self) -> np.ndarray:
        return self._sequence.get_all_frame_sizes()

    def get_all_object_bounding_box(self) -> np.ndarray:
        if not self._sequence.has_bounding_box():
            return np.full((len(self), 4), float('nan'), np.float64)
        return self._sequence.get_all_bounding_boxes()

    def get_all_object_existence_flag(self) -> Optional[np.ndarray]:
        if not self._sequence.has_bounding_box():
            return np.zeros((len(self), ), dtype=bool)
        return self._sequence.get_all_bounding_box_validity_flags() if self._sequence.has_bounding_box_validity_flag() else None

    def get_name(self) -> str:
        return self._sequence.get_name()

    def get_attribute(self, name: str) -> Any:
        return self._sequence.get_object_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._sequence.get_all_object_attribute_names()


class SOTSequence(TrackingDataset_Sequence):
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        self._sequence = sequence
        self._category_id = sequence.get_category_id() if sequence.has_category_id() else None

    def __getitem__(self, index: int) -> TrackingDataset_Frame:
        return SOTFrame(self._sequence[index], self._category_id)

    def __len__(self) -> int:
        return len(self._sequence)

    def get_track_by_index(self, index: int) -> TrackingDataset_Track:
        assert index == 0
        return SOTTrack(self._sequence, self._category_id)

    def get_track_by_id(self, id_: int) -> TrackingDataset_Track:
        assert id_ == 0
        return SOTTrack(self._sequence, self._category_id)

    def get_all_object_id(self) -> Tuple[int, ...]:
        return 0,

    def get_name(self) -> str:
        return self._sequence.get_name()

    def get_fps(self) -> Optional[float]:
        if not self._sequence.has_fps():
            return None
        return self._sequence.get_fps()

    def get_number_of_tracks(self) -> int:
        return 1

    def get_attribute(self, name: str) -> Any:
        return self._sequence.get_sequence_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._sequence.get_all_sequence_attribute_names()


class SOTDataset(TrackingDataset):
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> TrackingDataset_Sequence:
        return SOTSequence(self._dataset[index])

    def get_category_name_by_id(self, id_: int) -> Optional[str]:
        if not self._dataset.has_category_id_name_map():
            return None
        return self._dataset.get_category_name_by_id(id_)

    def get_all_category_id_name_map(self) -> Optional[Mapping[int, str]]:
        if not self._dataset.has_category_id_name_map():
            return None
        return self._dataset.get_category_id_name_map()

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
