from trackit.datasets.common.seed import BaseSeed


class Youtube_VIS_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('Youtube_VIS_PATH')
        super().__init__('Youtube-VIS', root_path, 'train', ('train',))

    def construct(self, constructor):
        from .Impl.YouTube_VIS import construct_Youtube_VIS
        construct_Youtube_VIS(constructor, self)
