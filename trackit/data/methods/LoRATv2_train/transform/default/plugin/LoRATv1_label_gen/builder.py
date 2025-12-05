def build_LoRATv1_label_generator(plugin_config: dict, config: dict):
    common_config = config['common']
    from . import BoxWithScoreMapLabelGenerator
    model_stride = common_config['model_stride']
    search_region_size = common_config['search_regions'][-1]['size']
    response_map_size = (search_region_size[0] // model_stride, search_region_size[1] // model_stride)
    generator = BoxWithScoreMapLabelGenerator(response_map_size, search_region_size)
    return generator, generator.collate
