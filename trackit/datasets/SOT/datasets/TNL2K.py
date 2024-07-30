from typing import Optional
from trackit.datasets.common.seed import BaseSeed


class TNL2K_Seed(BaseSeed):
    def __init__(self, root_path: Optional[str]=None, data_split='train'):
        if root_path is None:
            if data_split == 'train':
                root_path = self.get_path_from_config('TNL2K_TRAIN_PATH')
            if data_split == 'test':
                root_path = self.get_path_from_config('TNL2K_TEST_PATH')
        super().__init__('TNL2K', root_path, data_split, ('train', 'test'), 6)

    def construct(self, constructor):
        from .Impl.TNL2K import construct_TNL2K
        construct_TNL2K(constructor, self)
