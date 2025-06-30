import torch
from trackit.miscellanies.printing import pretty_format


def build_plugins(plugins_config: list[dict], config: dict, device: torch.device):
    print('plugins:\n' + pretty_format(plugins_config, indent_level=1))
    plugins = []
    for plugin_config in plugins_config:
        if plugin_config['type'] == 'template_foreground_indicating_mask_generation':
            from .template_foreground_indicating_mask_generation import TemplateFeatForegroundMaskGeneration
            plugins.append(TemplateFeatForegroundMaskGeneration(config['common']['template_size'],
                                                                config['common']['template_feat_size'],
                                                                device))
        else:
            raise ValueError('Unknown plugin type: {}'.format(plugin_config['type']))
    return plugins
