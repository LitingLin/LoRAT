from __future__ import annotations
from typing import Optional, Tuple, Mapping, Any
import numpy as np


class TrackingDataset:
    def get_name(self) -> Optional[str]:
        raise NotImplementedError()

    def get_full_name(self) -> Optional[str]:
        raise NotImplementedError()

    def get_data_split(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> TrackingDataset_Sequence:
        raise NotImplementedError()

    def get_category_name_by_id(self, id_: int) -> Optional[str]:
        raise NotImplementedError()

    def get_all_category_id_name_map(self) -> Optional[Mapping[int, str]]:
        raise NotImplementedError()

    def get_attribute(self, name: str) -> Any:
        raise NotImplementedError()

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        raise NotImplementedError()


class TrackingDataset_Sequence:
    def get_name(self) -> Optional[str]:
        raise NotImplementedError()

    def get_fps(self) -> Optional[float]:
        raise NotImplementedError()

    def get_number_of_tracks(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> TrackingDataset_Frame:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def get_track_by_index(self, index: int) -> TrackingDataset_Track:
        raise NotImplementedError()

    def get_track_by_id(self, id_: int) -> TrackingDataset_Track:
        raise NotImplementedError()

    def get_all_object_id(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_attribute(self, name: str) -> Any:
        raise NotImplementedError()

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    def get_track_iterator(self):
        for i in range(self.get_number_of_tracks()):
            yield self.get_track_by_index(i)


class TrackingDataset_Track:
    def get_name(self) -> Optional[str]:
        raise NotImplementedError()

    def get_object_id(self) -> int:
        raise NotImplementedError()

    def get_category_id(self) -> Optional[int]:
        raise NotImplementedError()

    def get_all_frame_size(self) -> np.ndarray:
        raise NotImplementedError()

    def get_all_object_bounding_box(self) -> np.ndarray:
        '''
        :return: bounding box should be with dtype np.float64
        '''
        raise NotImplementedError()

    def get_all_object_existence_flag(self) -> Optional[np.ndarray]:
        '''
        :return: optional, None indicates the object exists in all frames
        '''
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> TrackingDataset_FrameInTrack:
        raise NotImplementedError()

    def get_attribute(self, name: str) -> Any:
        raise NotImplementedError()

    def get_all_attribute_name(self) -> Tuple[str, ...]:
        raise NotImplementedError()


class _TrackingDataset_BaseFrame:
    def get_frame_size(self) -> np.ndarray:
        raise NotImplementedError()

    def get_frame(self) -> np.ndarray:
        # (H, W, C)
        # C=3, RGB
        raise NotImplementedError()

    def get_frame_file_path(self) -> Optional[str]:
        raise NotImplementedError()

    def get_frame_index(self) -> int:
        raise NotImplementedError()

    def is_file_backed(self) -> bool:
        raise NotImplementedError()

    def get_frame_attribute(self, name: str) -> Any:
        raise NotImplementedError()

    def get_all_frame_attribute_name(self) -> Tuple[str, ...]:
        raise NotImplementedError()


class TrackingDataset_Frame(_TrackingDataset_BaseFrame):
    def get_number_of_objects(self) -> int:
        raise NotImplementedError()

    def get_all_object_id(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_object_by_index(self, index: int) -> TrackingDataset_Object:
        raise NotImplementedError()

    def get_object_by_id(self, id_) -> TrackingDataset_Object:
        raise NotImplementedError()


class TrackingDataset_Object:
    def get_object_id(self) -> int:
        raise NotImplementedError()

    def get_category_id(self) -> Optional[int]:
        raise NotImplementedError()

    def get_bounding_box(self) -> np.ndarray:
        '''
        :return: bounding box should be with dtype np.float64
        '''
        raise NotImplementedError()

    def get_existence_flag(self) -> bool:
        raise NotImplementedError()

    def get_object_attribute(self, name: str) -> Any:
        raise NotImplementedError()

    def get_all_object_attribute_name(self) -> Tuple[str, ...]:
        raise NotImplementedError()


class TrackingDataset_FrameInTrack(_TrackingDataset_BaseFrame, TrackingDataset_Object):
    def get_frame_index_in_sequence(self) -> int:
        raise NotImplementedError()
