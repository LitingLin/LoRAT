from trackit.datasets.common.seed import BaseSeed


class COCO_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=('train', 'val'), version: int = 2014, include_crowd=False):
        flags = []
        if version == 2014:
            flags.append('2014')
            if root_path is None:
                root_path = self.get_path_from_config('COCO_2014_PATH')
        elif version == 2017:
            flags.append('2017')
            if root_path is None:
                root_path = self.get_path_from_config('COCO_2017_PATH')
        else:
            raise Exception
        if not include_crowd:
            flags.append('nocrowd')
        super().__init__('COCO', root_path, data_split, ('train', 'val'), flags)
        self.coco_version = version
        self.include_crowd = include_crowd

    def construct(self, constructor):
        from .impl.COCO import construct_COCO
        construct_COCO(constructor, self)
