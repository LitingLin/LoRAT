import os
import itertools
from typing import Mapping
from functools import partial
import yaml
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

from .mixin_rule import apply_mixin_rule

class CustomLoader(Loader):
    def __init__(self, stream, const_values: Mapping | None = None):
        self._root = os.path.split(stream.name)[0]
        self.yaml_constructors['!include'] = self.__class__.include
        self.yaml_constructors['!concat'] = self.__class__.concat
        self.yaml_constructors['!combine'] = self.__class__.combine
        self._const_values = const_values
        if const_values is not None:
            self.yaml_constructors['!const'] = self.__class__.const

        super(CustomLoader, self).__init__(stream)

    def include(self, node):
        if not isinstance(node, yaml.ScalarNode):
            raise TypeError('!include tag must be a scalar')
        filename = os.path.join(self._root, self.construct_scalar(node))

        return load_yaml(filename, self._const_values)

    def const(self, node):
        if isinstance(node, yaml.ScalarNode):
            const_name = self.construct_scalar(node)
            return self._const_values[const_name]
        elif isinstance(node, yaml.SequenceNode):
            const_name = self.construct_sequence(node)
            return self._const_values[const_name]
        else:
            raise TypeError('!const tag must be a scalar or sequence')

    def concat(self, node):
        if not isinstance(node, yaml.SequenceNode):
            raise TypeError('!concat tag must be a sequence')
        constructed_subnodes = [self.construct_object(i, deep=True) for i in node.value]
        if all(isinstance(i, (str, int, float)) for i in constructed_subnodes):
            return ''.join(str(i) for i in constructed_subnodes)
        elif all(isinstance(i, list) for i in constructed_subnodes):
            return list(itertools.chain(*constructed_subnodes))
        else:
            raise TypeError('!concat tag must be a sequence of scalar or sequence')

    def combine(self, node):
        if not isinstance(node, yaml.MappingNode):
            raise TypeError('!combine tag must be a mapping')

        # Construct the mapping for __base__ and __mixin__.
        # Using deep=True ensures that any !include tags within __base__ and __mixin__
        # are already resolved to their loaded content (dictionaries/lists).
        combine_spec = self.construct_mapping(node, deep=True)

        base_config = combine_spec['__base__']
        mixin_rules_list = combine_spec.get('__mixin__', [])
        for key in combine_spec:
            if key not in ('__base__', '__mixin__'):
                raise TypeError('!combine tag must be a mapping node with __base__ and __mixin__')

        # Apply each set of mixin rules to the base configuration
        for mixin_rules in mixin_rules_list:
            if isinstance(mixin_rules, (list, tuple)):
                for mixin_rule in mixin_rules:
                    apply_mixin_rule(mixin_rule, base_config)
            elif isinstance(mixin_rules, Mapping):
                apply_mixin_rule(mixin_rules, base_config)
            else:
                raise TypeError('__mixin__ must be a mapping or a sequence of mappings')

        return base_config

def load_yaml(path: str, const_values: Mapping | None = None):
    loader_cls = CustomLoader
    if const_values is not None:
        loader_cls = partial(CustomLoader, const_values=const_values)
    with open(path, 'rb') as f:
        object_ = yaml.load(f, Loader=loader_cls)
    return object_
