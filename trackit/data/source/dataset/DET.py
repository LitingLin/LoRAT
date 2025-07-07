from __future__ import annotations
from trackit.datasets.DET.specialization.memory_mapped.dataset import DetectionDataset_MemoryMapped, DetectionDatasetImage_MemoryMapped, DetectionDatasetObject_MemoryMapped
from trackit.data.source import *
from trackit.miscellanies.image.io import *


class DETImageWithObject(TrackingDataset_Track, TrackingDataset_FrameInTrack):
    def __init__(self, image: DetectionDatasetImage_MemoryMapped, object_index: int, object_: DetectionDatasetObject_MemoryMapped):
        self._image = image
        self._object_index = object_index
        self._object = object_

    def is_file_backed(self) -> bool:
        return True

    def get_object_id(self) -> int:
        return self._object_index

    def get_category_id(self) -> Optional[int]:
        return self._object.get_category_id() if self._object.has_category_id() else None

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> TrackingDataset_FrameInTrack:
        assert index == 0
        return self

    def get_frame_size(self) -> np.ndarray:
        return self._image.get_image_size()

    def get_frame(self) -> np.ndarray:
        return read_image_with_auto_retry(self._image.get_image_path())

    def get_frame_file_path(self) -> str:
        return self._image.get_image_path()

    def get_frame_index(self) -> int:
        return 0

    def get_frame_index_in_sequence(self) -> int:
        return 0

    def get_bounding_box(self) -> np.ndarray:
        if not self._object.has_bounding_box():
            return np.full((4,), float('nan'), np.float64)
        return self._object.get_bounding_box().astype(np.float64)

    def get_existence_flag(self) -> bool:
        if not self._object.has_bounding_box():
            return False
        return self._object.get_bounding_box_validity_flag().item() if self._object.has_bounding_box_validity_flag() else True

    def get_name(self) -> Optional[str]:
        return None

    def get_all_frame_size(self) -> np.ndarray:
        return np.expand_dims(self._image.get_image_size(), 0)

    def get_all_object_bounding_box(self) -> Optional[np.ndarray]:
        if not self._object.has_bounding_box():
            return np.full((1, 4), float('nan'), np.float64)
        return np.expand_dims(self._object.get_bounding_box().astype(np.float64), 0)

    def get_all_object_existence_flag(self) -> Optional[np.ndarray]:
        if not self._object.has_bounding_box():
            return np.zeros((1,), dtype=bool)  # False
        if not self._object.has_bounding_box_validity_flag():
            return None
        return self._object.get_bounding_box_validity_flag().reshape((1,))

    def get_attribute(self, name: str) -> Any:
        return self._image.get_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._image.get_all_attribute_name()

    def get_object_attribute(self, name: str) -> Any:
        return self._object.get_attribute(name)

    def get_all_object_attribute_name(self) -> Tuple[str, ...]:
        return self._object.get_all_attribute_name()

    def get_frame_attribute(self, name: str) -> Any:
        return self._image.get_attribute(name)

    def get_all_frame_attribute_name(self) -> Tuple[str, ...]:
        return self._image.get_all_attribute_name()


class DETImage(TrackingDataset_Frame):
    def __init__(self, frame: DetectionDatasetImage_MemoryMapped):
        self._image = frame

    def is_file_backed(self) -> bool:
        return True

    def get_frame_size(self) -> np.ndarray:
        return np.array(self._image.get_image_size(), dtype=int)

    def get_frame(self) -> np.ndarray:
        return read_image_with_auto_retry(self._image.get_image_path())

    def get_frame_file_path(self) -> str:
        return self._image.get_image_path()

    def get_number_of_objects(self) -> int:
        return len(self._image)

    def get_all_object_id(self) -> Tuple[int, ...]:
        return tuple(range(len(self._image)))

    def get_object_by_index(self, index: int) -> TrackingDataset_Object:
        return DETImageWithObject(self._image, index, self._image[index])

    def get_object_by_id(self, id_: int) -> TrackingDataset_Object:
        return self.get_object_by_index(id_)

    def get_frame_index(self) -> int:
        return 0

    def get_frame_attribute(self, name: str) -> Any:
        return self._image.get_attribute(name)

    def get_all_frame_attribute_name(self) -> Tuple[str, ...]:
        return self._image.get_all_attribute_name()


class DETSequence(TrackingDataset_Sequence):
    def __init__(self, image: DetectionDatasetImage_MemoryMapped):
        self._image = image

    def get_name(self) -> Optional[str]:
        return None

    def get_fps(self) -> Optional[float]:
        return None

    def get_number_of_tracks(self) -> int:
        return len(self._image)

    def __getitem__(self, index: int) -> TrackingDataset_Frame:
        if index != 0:
            raise IndexError("DET dataset only supports a single frame per image.")
        return DETImage(self._image)

    def __len__(self) -> int:
        return 1

    def get_track_by_index(self, index: int) -> TrackingDataset_Track:
        return DETImageWithObject(self._image, index, self._image[index])

    def get_track_by_id(self, id_: int) -> TrackingDataset_Track:
        return self.get_track_by_index(id_)

    def get_all_object_id(self) -> Tuple[int, ...]:
        return tuple(range(len(self._image)))

    def get_attribute(self, name: str) -> Any:
        return self._image.get_attribute(name)

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        return self._image.get_all_attribute_name()


class DETDataset(TrackingDataset):
    def __init__(self, dataset: DetectionDataset_MemoryMapped):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> DETSequence:
        return DETSequence(self._dataset[index])

    def get_category_name_by_id(self, id_: int) -> Optional[str]:
        return self._dataset.get_category_name_by_id(id_) if self._dataset.has_category_id_name_map() else None

    def get_all_category_id_name_map(self) -> Optional[Mapping[int, str]]:
        return self._dataset.get_category_id_name_map() if self._dataset.has_category_id_name_map() else None

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
