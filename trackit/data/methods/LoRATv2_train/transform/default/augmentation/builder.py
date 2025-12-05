from .pipeline import AugmentationPipeline, AugmentationConfig


def build_augmentation_pipeline(augmentation_pipline_config: list):
    pipelines = []
    for augmentation_config in augmentation_pipline_config:
        if augmentation_config['type'] == 'gray_scale':
            from .gray_scale import GrayScaleAugmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   _build_range_selector(augmentation_config.get('target_selector', None)),
                                   GrayScaleAugmentation(augmentation_config['probability']),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'horizontal_flip':
            from .horizontal_flip import HorizontalFlipAugmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   _build_range_selector(augmentation_config.get('target_selector', None)),
                                   HorizontalFlipAugmentation(augmentation_config['probability']),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'color_jitter':
            from .color_jitter import ColorJitter

            brightness_factor = augmentation_config['brightness']
            contrast_factor = augmentation_config['contrast']
            saturation_factor = augmentation_config['saturation']

            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   _build_range_selector(augmentation_config.get('target_selector', None)),
                                   ColorJitter(brightness_factor, contrast_factor, saturation_factor),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'DeiT_3_aug':
            from .deit_3_augmentation import DeiT3Augmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   _build_range_selector(augmentation_config.get('target_selector', None)),
                                   DeiT3Augmentation(),
                                   augmentation_config.get('joint', True)))
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_config['type']}")
    return AugmentationPipeline(pipelines)


def _build_range_selector(range_selector_config: None | int | dict):
    if range_selector_config is None:
        return None
    elif isinstance(range_selector_config, int):
        return slice(range_selector_config, range_selector_config + 1)
    elif isinstance(range_selector_config, dict):
        return slice(range_selector_config['start'], range_selector_config['stop'], range_selector_config['step'])
    else:
        raise ValueError(f"Unknown range selector type: {type(range_selector_config)}")
