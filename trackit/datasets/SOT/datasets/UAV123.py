from trackit.datasets.common.seed import BaseSeed


class UAV123_Seed(BaseSeed):
    def __init__(self, root_path: str=None, build_UAV20L=False):
        if root_path is None:
            root_path = self.get_path_from_config('UAV123_PATH')
        self.build_UAV20L = build_UAV20L
        if build_UAV20L:
            dataset_name = 'UAV20L'
        else:
            dataset_name = 'UAV123'
        super().__init__(dataset_name, root_path, None, (), 4)

    def construct(self, constructor):
        from .Impl.UAV123 import construct_UAV123
        construct_UAV123(constructor, self)
