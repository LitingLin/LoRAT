from trackit.datasets.common.seed import BaseSeed


class LaSOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'test')):
        if root_path is None:
            root_path = self.get_path_from_config('LaSOT_PATH')
        super().__init__('LaSOT', root_path, data_split, ('train', 'test'), version=2)

    def construct(self, constructor):
        from .Impl.LaSOT import construct_LaSOT
        construct_LaSOT(constructor, self)


def get_LaSOT_sequence_attributes():
    from .Impl.LaSOT import LaSOTAttributes
    return LaSOTAttributes
