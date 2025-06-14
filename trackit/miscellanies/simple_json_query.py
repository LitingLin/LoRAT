# A very simple query language for json / yaml
#
#  following Pytorch nn.Module naming convention
#  JsonPath or Json Query is too complex for our use cases.
#
# examples:
#   config = {'a': {'b': {'c': 1}}}
#   simple_json_query_get(config, 'a.b.c')
#   >>> 1
#   simple_json_query_set(config, 'a.b.c', 2)
#   config
#   >>> {'a': {'b': {'c': 2}}}
#   config = {'a': [{'b': {'c': 1}}, {'b': {'c': 2}}]}
#   simple_json_query_get(config, 'a.0.b.c')
#   >>> 1
#   simple_json_query_set(config, 'a.0.b.c', 3)
#   config
#   >>> {'a': [{'b': {'c': 3}}, {'b': {'c': 2}}]}


def _split_path(path: str):
    """
    Split a dot-separated path into keys. Convert digit-only segments to integers.
    Example:
        "a.b.0.c" -> ["a", "b", 0, "c"]
    """
    parts = path.split('.')
    return [int(p) if p.isdigit() else p for p in parts]


def _navigate(obj, path: str):
    """
    Navigate through obj according to path and return the target object and the final key.
    This function returns the container that holds the final key, and the key itself.
    For example:
        obj = {'a': {'b': {'c': 1}}}
        _navigate(obj, 'a.b.c') -> ({'c': 1}, 'c')

        obj = {'a': [{'b': 0}, {'c': 1}]}
        _navigate(obj, 'a.1.c') -> ({'c': 1}, 'c')
    """
    keys = _split_path(path)
    if not keys:
        raise ValueError("Path cannot be empty.")

    # Traverse to the parent of the final key
    current = obj
    for key in keys[:-1]:
        current = current[key]

    return current, keys[-1]


def simple_json_query_get(json_object, path: str):
    """
    Get a value from a nested structure by following a dot-separated path.

    Example:
        data = {'x': {'y': {'z': 42}}}
        simple_json_query_get(data, 'x.y.z') -> 42
    """
    keys = _split_path(path)
    current = json_object
    for key in keys:
        current = current[key]
    return current


def simple_json_query_set(json_object, path: str, value):
    """
    Set a value in a nested structure at a given dot-separated path.

    Example:
        data = {'x': {'y': {'z': 42}}}
        simple_json_query_set(data, 'x.y.z', 100)
        # data -> {'x': {'y': {'z': 100}}}
    """
    parent, final_key = _navigate(json_object, path)
    parent[final_key] = value


def simple_json_query_retain(json_object, path: str, keys_to_keep):
    """
    Retain only specified keys in a dictionary or retain items at specified indices in a list at a given path.
    Remove all other keys or items.

    :param json_object: The JSON-like object (dict or list).
    :param path: Dot-separated path to the target.
    :param keys_to_keep:
        - If the target is a **dict**, a single key or an iterable of keys to retain.
        - If the target is a **list**, a single integer index or an iterable of indices to retain.

    Example:
        # Retaining keys in a dictionary
        data = {'config': {'a': 1, 'b': 2, 'c': 3}}
        simple_json_query_retain(data, 'config', ['a', 'c'])
        # data -> {'config': {'a': 1, 'c': 3}}

        # Retaining items at specific indices in a list
        data = {'items': ['apple', 'banana', 'cherry', 'date']}
        simple_json_query_retain(data, 'items', [0, 2])
        # data -> {'items': ['apple', 'cherry']}
    """
    if not isinstance(keys_to_keep, (list, tuple)):
        keys_to_keep = [keys_to_keep]

    target = simple_json_query_get(json_object, path)

    if isinstance(target, dict):
        for key in list(target.keys()):
            if key not in keys_to_keep:
                del target[key]
    elif isinstance(target, list):
        if not all(isinstance(k, int) for k in keys_to_keep):
            raise TypeError("When the target is a list, keys_to_keep must be integers representing indices.")
        target[:] = [item for i, item in enumerate(target) if i in keys_to_keep]
    else:
        raise TypeError(f"Invalid target type: {type(target)}")


def simple_json_query_retain_by_value(json_object, path: str, values_to_keep):
    """
    Retain only items in a dictionary or list based on their values at a given path.
    Remove all other items.

    :param json_object: The JSON-like object (dict or list).
    :param path: Dot-separated path to the target.
    :param values_to_keep:
        - If the target is a **dict**, retain key-value pairs where the value is in `values_to_keep`.
        - If the target is a **list**, retain items that are in `values_to_keep`.

    Example:
        # Retaining dictionary items by value
        data = {'config': {'a': 1, 'b': 2, 'c': 3}}
        simple_json_query_retain_by_value(data, 'config', [1, 3])
        # data -> {'config': {'a': 1, 'c': 3}}

        # Retaining list items by value
        data = {'numbers': [1, 2, 3, 4, 2]}
        simple_json_query_retain_by_value(data, 'numbers', [2, 4])
        # data -> {'numbers': [2, 4, 2]}
    """
    if not isinstance(values_to_keep, (list, tuple)):
        values_to_keep = [values_to_keep]

    target = simple_json_query_get(json_object, path)

    if isinstance(target, dict):
        for key in list(target.keys()):
            if target[key] not in values_to_keep:
                del target[key]
    elif isinstance(target, list):
        target[:] = [item for item in target if item in values_to_keep]
    else:
        raise TypeError(f"Invalid target type: {type(target)}")


def simple_json_query_remove(json_object, path: str, keys):
    """
    Remove specified keys from a dictionary or remove items at specified indices from a list at a given path.

    :param json_object: The JSON-like object (dict or list).
    :param path: Dot-separated path to the target.
    :param keys:
        - If the target is a **dict**, a single key or an iterable of keys to remove.
        - If the target is a **list**, a single integer index or an iterable of indices to remove.

    Example:
        # Removing keys from a dictionary
        data = {'settings': {'a': 1, 'b': 2, 'c': 3}}
        simple_json_query_remove(data, 'settings', 'b')
        # data -> {'settings': {'a': 1, 'c': 3}}

        # Removing multiple keys from a dictionary
        data = {'settings': {'a': 1, 'b': 2, 'c': 3}}
        simple_json_query_remove(data, 'settings', ['a', 'c'])
        # data -> {'settings': {'b': 2}}

        # Removing an item by index from a list
        data = {'items': ['apple', 'banana', 'cherry']}
        simple_json_query_remove(data, 'items', 1)
        # data -> {'items': ['apple', 'cherry']}

        # Removing multiple items by indices from a list
        data = {'items': ['apple', 'banana', 'cherry', 'date']}
        simple_json_query_remove(data, 'items', [0, 2])
        # data -> {'items': ['banana', 'date']}
    """
    if not isinstance(keys, (list, tuple)):
        keys = [keys]

    target = simple_json_query_get(json_object, path)

    if isinstance(target, dict):
        for k in keys:
            if k in target:
                del target[k]
    elif isinstance(target, list):
        if not all(isinstance(k, int) for k in keys):
            raise TypeError("When the target is a list, keys must be integers representing indices.")
        target[:] = [item for i, item in enumerate(target) if i not in keys]
    else:
        raise TypeError(f"Invalid target type: {type(target)}")


def simple_json_query_remove_by_value(json_object, path: str, values):
    """
    Remove specified key-value pairs from a dictionary or remove items by value from a list at a given path.

    :param json_object: The JSON-like object (dict or list).
    :param path: Dot-separated path to the target.
    :param values:
        - If the target is a **dict**, remove key-value pairs where the value is in `values`.
        - If the target is a **list**, remove items that are in `values`.

    Example:
        # Removing key-value pairs from a dictionary based on value
        data = {'settings': {'a': 1, 'b': 2, 'c': 3}}
        simple_json_query_remove_by_value(data, 'settings', 2)
        # data -> {'settings': {'a': 1, 'c': 3}}

        # Removing multiple key-value pairs from a dictionary based on values
        data = {'settings': {'a': 1, 'b': 2, 'c': 3, 'd': 2}}
        simple_json_query_remove_by_value(data, 'settings', [2, 3])
        # data -> {'settings': {'a': 1}}

        # Removing values from a list
        data = {'numbers': [1, 2, 3, 4, 2]}
        simple_json_query_remove_by_value(data, 'numbers', 2)
        # data -> {'numbers': [1, 3, 4]}

        # Removing multiple values from a list
        data = {'numbers': [1, 2, 3, 4, 2, 5]}
        simple_json_query_remove_by_value(data, 'numbers', [2, 4])
        # data -> {'numbers': [1, 3, 5]}
    """
    if not isinstance(values, (list, tuple)):
        values = [values]

    target = simple_json_query_get(json_object, path)

    if isinstance(target, dict):
        for k, v in list(target.items()):
            if v in values:
                del target[k]
    elif isinstance(target, list):
        target[:] = [item for item in target if item not in values]
    else:
        raise TypeError(f"Invalid target type: {type(target)}")


def simple_json_query_merge(json_object, path: str, value):
    """
    Merge a given value into a list or dict at a specified path.

    If the target is a list:
      - `value` should be an iterable (non-string) to extend the list.
    If the target is a dict:
      - `value` should be a dictionary to update the target.

    Example:
        # Merging into a list
        data = {'items': [1, 2]}
        simple_json_query_merge(data, 'items', [3, 4])
        # data -> {'items': [1, 2, 3, 4]}

        # Merging into a dict
        data = {'info': {'a': 1}}
        simple_json_query_merge(data, 'info', {'b': 2})
        # data -> {'info': {'a': 1, 'b': 2}}
    """
    target = simple_json_query_get(json_object, path)
    if isinstance(target, list):
        target.extend(value)
    elif isinstance(target, dict):
        target.update(value)
    else:
        raise TypeError(f"Invalid target type: {type(target)}")


def simple_json_query_append(json_object, path: str, value):
    """
    Append a value to a list or concatenate a string at the given path.

    If the target is a list:
      - Append the given value.
    If the target is a string:
      - Concatenate the given value (also a string).

    Example:
        # Appending to a list
        data = {'fruits': ['apple', 'banana']}
        simple_json_query_append(data, 'fruits', 'cherry')
        # data -> {'fruits': ['apple', 'banana', 'cherry']}

        # Concatenating strings
        data = {'greeting': 'Hello'}
        simple_json_query_append(data, 'greeting', ' World!')
        # data -> {'greeting': 'Hello World!'}
    """
    target = simple_json_query_get(json_object, path)
    if isinstance(target, list):
        target.append(value)
    elif isinstance(target, str):
        if not isinstance(value, str):
            raise ValueError("When appending to a string, value must also be a string.")
        new_value = target + value
        simple_json_query_set(json_object, path, new_value)
    else:
        raise TypeError("simple_json_query_append requires the target to be a list or a string.")
