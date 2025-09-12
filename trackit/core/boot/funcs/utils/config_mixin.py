import os

from trackit.core.runtime.utils.custom_yaml_loader import load_yaml
from trackit.core.runtime.utils.mixin_rule import apply_mixin_rule
from trackit.core.runtime.global_constant import get_global_constant


def apply_static_mixin_rules(mixin_rules, config):
    for static_modification_rule in mixin_rules:
        apply_mixin_rule(static_modification_rule, config, None)


def apply_mixin_rules(mixin_rules, config, dynamic_values):
    if 'static' in mixin_rules:
        for static_modification_rule in mixin_rules['static']:
            apply_mixin_rule(static_modification_rule, config, None)
    if 'dynamic' in mixin_rules:
        for dynamic_parameter_name, dynamic_modification_rule in mixin_rules['dynamic'].items():
            if 'type' in dynamic_modification_rule and dynamic_modification_rule['type'] == 'bool':
                if bool(dynamic_values[dynamic_parameter_name]):
                    apply_mixin_rule(dynamic_modification_rule, config, None)
            else:
                apply_mixin_rule(dynamic_modification_rule, config, dynamic_values[dynamic_parameter_name])


def load_static_mixin_config(args):
    configs = []
    for mixin_config in args.mixin_config:
        mixin_config = mixin_config + '.yaml'
        if mixin_config.startswith('/' or '\\'):
            config_path = os.path.join(args.config_path, mixin_config[1:])
            assert os.path.exists(config_path), 'mixin config not found: {}'.format(config_path)
        else:
            candidate_paths = [os.path.join(args.config_path, args.method_name, args.config_name, 'mixin', mixin_config),
                               os.path.join(args.config_path, args.method_name, '_mixin', mixin_config),
                               os.path.join(args.config_path, '_mixin', mixin_config)]

            found = False
            for config_path in candidate_paths:
                if os.path.exists(config_path):
                    found = True
                    break
            assert found, 'mixin config not found in paths: {}'.format(candidate_paths)
        config = load_yaml(config_path, get_global_constant())
        configs.append(config)
    return configs


def load_static_mixin_config_and_apply_rules(args, config):
    mixin_configs = load_static_mixin_config(args)
    for mixin_config in mixin_configs:
        apply_static_mixin_rules(mixin_config, config)
