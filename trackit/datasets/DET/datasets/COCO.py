from trackit.datasets.common.seed import BaseSeed


class COCO_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=('train', 'val'), version: int = 2014, include_crowd=False):
        if version == 2014:
            name = 'COCO2014'
            if root_path is None:
                root_path = self.get_path_from_config('COCO_2014_PATH')
        elif version == 2017:
            name = 'COCO2017'
            if root_path is None:
                root_path = self.get_path_from_config('COCO_2017_PATH')
        else:
            raise Exception
        if not include_crowd:
            name += '-nocrowd'
        super().__init__(name, root_path, data_split, ('train', 'val'), 2)
        self.coco_version = version
        self.include_crowd = include_crowd

    def construct(self, constructor):
        from .impl.COCO import construct_COCO
        construct_COCO(constructor, self)
