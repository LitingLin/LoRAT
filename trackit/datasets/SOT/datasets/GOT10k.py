from trackit.datasets.common.seed import BaseSeed


class GOT10k_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split='train', data_specs=None, absense_cover_level=None):
        if root_path is None:
            root_path = self.get_path_from_config('GOT10k_PATH')
        self.data_specs = data_specs
        self.absense_cover_level = absense_cover_level
        flags = []
        if data_specs is not None:
            flags.append(data_specs)
        if absense_cover_level is not None:
            flags.append(f"absense_cover_level-{absense_cover_level}")
        super().__init__('GOT-10k', root_path, data_split, ('train', 'val', 'test'), flags)

    def construct(self, constructor):
        from .Impl.GOT10k import construct_GOT10k
        construct_GOT10k(constructor, self)
