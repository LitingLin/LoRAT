import os
from ..utils.custom_yaml_loader import load_yaml
from ...funcs.mixin import load_static_mixin_config_and_apply_rules


def load_config(runtime_vars):
    config_path = os.path.join(runtime_vars.config_path, runtime_vars.method_name, runtime_vars.config_name, 'config.yaml')
    config = load_yaml(config_path)
    if runtime_vars.mixin_config is not None:
        load_static_mixin_config_and_apply_rules(runtime_vars, config)
    return config
