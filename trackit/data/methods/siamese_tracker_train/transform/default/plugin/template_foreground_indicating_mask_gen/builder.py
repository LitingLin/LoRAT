from . import TemplateFeatMaskGenerator, template_feat_mask_data_collator


def build_template_feat_foreground_mask_generator(config: dict):
    common_config = config['common']
    return (TemplateFeatMaskGenerator(common_config['template_size'], common_config['template_feat_size']),
            template_feat_mask_data_collator)
