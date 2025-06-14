from trackit.datasets.common.seed import BaseSeed


class ILSVRC_DET_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val')):
        if root_path is None:
            root_path = self.get_path_from_config('ILSVRC_DET_PATH')
        super().__init__('ILSVRC_DET', root_path, data_split, ('train', 'val'))

    def construct(self, constructor):
        from .impl.ILSVRC_DET import construct_ILSVRC_DET
        construct_ILSVRC_DET(constructor, self)
