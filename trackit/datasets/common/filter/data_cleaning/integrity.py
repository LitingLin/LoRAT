from .._common import _BaseFilter


class DataCleaning_Integrity(_BaseFilter):
    def __init__(self, remove_zero_annotation_objects: bool = False, remove_zero_annotation_image: bool = False,
                 remove_zero_annotation_video_head_tail: bool = False, remove_invalid_image: bool = True):
        self.remove_zero_annotation_objects = remove_zero_annotation_objects
        self.remove_zero_annotation_image = remove_zero_annotation_image
        self.remove_zero_annotation_video_head_tail = remove_zero_annotation_video_head_tail
        self.remove_invalid_image = remove_invalid_image
