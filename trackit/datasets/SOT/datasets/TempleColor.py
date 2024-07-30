from trackit.datasets.common.seed import BaseSeed


class TempleColor_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('TempleColor_PATH')
        super().__init__('TempleColor-128', root_path, None, (), 1)

    def construct(self, constructor):
        from .Impl.TempleColor import construct_TempleColor
        construct_TempleColor(constructor, self)
