from typing import Iterable
from trackit.datasets.common.seed import BaseSeed


class TrackingNet_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split='train', enable_set_ids: Iterable[int]=None, sequence_category_mapping_file_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('TrackingNet_PATH')
        name = 'TrackingNet'
        if enable_set_ids is not None:
            name += '-'
            name += '_'.join([str(v) for v in enable_set_ids])
        super().__init__(name, root_path, data_split, ('train', 'test'), 3)
        self.sequence_category_mapping_file_path = sequence_category_mapping_file_path
        self.enable_set_ids = enable_set_ids

    def construct(self, constructor):
        from .Impl.TrackingNet import construct_TrackingNet
        construct_TrackingNet(constructor, self)
