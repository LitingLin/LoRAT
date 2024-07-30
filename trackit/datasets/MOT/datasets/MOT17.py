from trackit.datasets.common.seed import BaseSeed


class MOT17_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('MOT17_PATH')
        super().__init__('MOT17', root_path, 'train', ('train',), 1)

    def construct(self, constructor):
        from .Impl.MOT17 import construct_MOT17
        construct_MOT17(constructor, self)
