from trackit.datasets.common.seed import BaseSeed


class MOT20_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('MOT20_PATH')
        super().__init__('MOT20', root_path, 'train', ('train',))

    def construct(self, constructor):
        from .Impl.MOT20 import construct_MOT20
        construct_MOT20(constructor, self)
