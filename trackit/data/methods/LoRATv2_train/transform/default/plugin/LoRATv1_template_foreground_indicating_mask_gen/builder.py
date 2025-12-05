def build_LoRATv1_template_feat_foreground_mask_generator(plugin_config: dict, config: dict):
    common_config = config['common']
    from . import TemplateFeatMaskGenerator
    template_size = common_config['templates'][0]['size']
    model_stride = common_config['model_stride']
    template_feat_size = (template_size[0] // model_stride, template_size[1] // model_stride)
    generator = TemplateFeatMaskGenerator(template_size, template_feat_size)
    return generator, generator.collate
