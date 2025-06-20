from trackit.datasets.common.seed import BaseSeed


class LaSOT_Extension_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('LaSOT_Extension_PATH')
        super().__init__('LaSOT_Extension', root_path, version=2)

    def construct(self, constructor):
        from .Impl.LaSOT_Extension import construct_LaSOT_Extension
        construct_LaSOT_Extension(constructor, self)


def get_LaSOT_extension_sequence_attributes():
    from .Impl.LaSOT_Extension import LaSOTExtensionAttributes
    return LaSOTExtensionAttributes
