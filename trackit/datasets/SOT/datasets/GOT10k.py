from trackit.datasets.common.seed import BaseSeed


class GOT10k_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split='train', data_specs=None):
        if root_path is None:
            root_path = self.get_path_from_config('GOT10k_PATH')
        self.data_specs = data_specs
        name = 'GOT-10k'
        if data_specs is not None:
            name += '-'
            name += data_specs
        super().__init__(name, root_path, data_split, ('train', 'val', 'test'), 2)

    def construct(self, constructor):
        from .Impl.GOT10k import construct_GOT10k
        construct_GOT10k(constructor, self)
