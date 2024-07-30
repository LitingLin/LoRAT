from trackit.datasets.common.seed import BaseSeed


class ImageNet_21k_Seed(BaseSeed):
    def __init__(self, root_path: str=None, skip_image_file_attributes=False):
        if root_path is None:
            root_path = self.get_path_from_config('ImageNet-21k_PATH')
        name = 'ImageNet-21k'
        if skip_image_file_attributes:
            name += '_skip_image_file_attributes'
        self.skip_image_file_attributes = skip_image_file_attributes
        super().__init__(name, root_path, None, (), 1)

    def construct(self, constructor):
        from .Impl.ImageNet_21k import construct_ImageNet_21k
        construct_ImageNet_21k(constructor, self)
