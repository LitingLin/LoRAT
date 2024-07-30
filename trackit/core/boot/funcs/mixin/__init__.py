import os
from .operator import json_query_replace, json_query_retain, json_query_remove, json_query_append, json_query_merge
from ..utils.custom_yaml_loader import load_yaml


def _apply_mixin_rule(rule: dict, config, value, action=None):
    access_path = rule['path']

    if action is None:
        if 'action' not in rule:
            action = 'replace'
        else:
            action = rule['action']

    if value is None:
        value = rule['value']

    if action == 'replace':
        if isinstance(access_path, (list, tuple)):
            [json_query_replace(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            json_query_replace(config, access_path, value)
    elif action == 'append':
        if isinstance(access_path, (list, tuple)):
            [json_query_append(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            json_query_append(config, access_path, value)
    elif action == 'merge':
        if isinstance(access_path, (list, tuple)):
            [json_query_merge(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            json_query_merge(config, access_path, value)
    elif action == 'retain':
        if isinstance(access_path, (list, tuple)):
            [json_query_retain(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            json_query_retain(config, access_path, value)
    elif action == 'remove':
        if isinstance(access_path, (list, tuple)):
            [json_query_remove(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            json_query_remove(config, access_path, value)
    else:
        raise NotImplementedError(action)


def apply_static_mixin_rules(mixin_rules, config):
    for static_modification_rule in mixin_rules:
        _apply_mixin_rule(static_modification_rule, config, None)


def apply_mixin_rules(mixin_rules, config, dynamic_values):
    if 'static' in mixin_rules:
        for static_modification_rule in mixin_rules['static']:
            _apply_mixin_rule(static_modification_rule, config, None)
    if 'dynamic' in mixin_rules:
        for dynamic_parameter_name, dynamic_modification_rule in mixin_rules['dynamic'].items():
            if 'type' in dynamic_modification_rule and dynamic_modification_rule['type'] == 'bool':
                if bool(dynamic_values[dynamic_parameter_name]):
                    _apply_mixin_rule(dynamic_modification_rule, config, None)
            else:
                _apply_mixin_rule(dynamic_modification_rule, config, dynamic_values[dynamic_parameter_name])


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
        config = load_yaml(config_path)
        configs.append(config)
    return configs


def load_static_mixin_config_and_apply_rules(args, config):
    mixin_configs = load_static_mixin_config(args)
    for mixin_config in mixin_configs:
        apply_static_mixin_rules(mixin_config, config)
