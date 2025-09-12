from typing import Mapping

from trackit.miscellanies.simple_json_query import (simple_json_query_set, simple_json_query_retain,
                                                    simple_json_query_remove, simple_json_query_append,
                                                    simple_json_query_merge, simple_json_query_remove_by_value,
                                                    simple_json_query_retain_by_value)


def apply_mixin_rule(rule: Mapping, config, value = None):
    access_path = rule['path']

    if 'action' not in rule:
        action = 'set'
    else:
        action = rule['action']

    if value is None:
        value = rule['value']

    if action == 'set':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_set(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_set(config, access_path, value)
    elif action == 'append':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_append(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_append(config, access_path, value)
    elif action == 'merge':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_merge(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_merge(config, access_path, value)
    elif action == 'retain':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_retain(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_retain(config, access_path, value)
    elif action == 'remove':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_remove(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_remove(config, access_path, value)
    elif action == 'remove_by_value':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_remove_by_value(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_remove_by_value(config, access_path, value)
    elif action == 'retain_by_value':
        if isinstance(access_path, (list, tuple)):
            [simple_json_query_retain_by_value(config, sub_access_path, value) for sub_access_path in access_path]
        else:
            simple_json_query_retain_by_value(config, access_path, value)
    else:
        raise NotImplementedError(action)
