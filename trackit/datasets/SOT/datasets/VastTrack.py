from trackit.datasets.common.seed import BaseSeed


class VastTrack_Seed(BaseSeed):
    def __init__(self, root_path: str = None, data_split: str = 'train'):
        if root_path is None:
            if data_split == 'train':
                root_path = self.get_path_from_config('VastTrack_TRAIN_PATH')
            if data_split == 'test':
                root_path = self.get_path_from_config('VastTrack_TEST_PATH')
        super().__init__('VastTrack', root_path, data_split, ('train', 'test'), 1)

    def construct(self, constructor):
        from .Impl.VastTrack import construct_VastTrack
        construct_VastTrack(constructor, self)
