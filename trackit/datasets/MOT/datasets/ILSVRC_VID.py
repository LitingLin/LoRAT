from trackit.datasets.common.seed import BaseSeed


class ILSVRC_VID_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val')):
        if root_path is None:
            root_path = self.get_path_from_config('ILSVRC_VID_PATH')
        super().__init__('ILSVRC_VID', root_path, data_split, ('train', 'val'), 1)

    def construct(self, constructor):
        from .Impl.ILSVRC_VID import construct_ILSVRC_VID
        construct_ILSVRC_VID(constructor, self)
