import os
from ..utils.config_mixin import load_static_mixin_config_and_apply_rules
from trackit.core.runtime.utils.custom_yaml_loader import load_yaml
from trackit.core.runtime.global_constant import get_global_constant


def load_config(runtime_vars):
    config_path = os.path.join(runtime_vars.config_path, runtime_vars.method_name, runtime_vars.config_name, 'config.yaml')
    config = load_yaml(config_path, get_global_constant())
    if runtime_vars.mixin_config is not None:
        load_static_mixin_config_and_apply_rules(runtime_vars, config)
    return config
