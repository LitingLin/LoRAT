import os
import itertools
from typing import Mapping
from functools import partial
import yaml
from yaml import CSafeLoader as Loader


class CustomLoader(Loader):
    def __init__(self, stream, const_values: Mapping | None = None):
        self._root = os.path.split(stream.name)[0]
        self.yaml_constructors['!include'] = self.__class__.include
        self.yaml_constructors['!concat'] = self.__class__.concat
        if const_values is not None:
            self._const_values = const_values
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


def load_yaml(path: str, const_values: Mapping | None = None):
    loader_cls = CustomLoader
    if const_values is not None:
        loader_cls = partial(CustomLoader, const_values=const_values)
    with open(path, 'rb') as f:
        object_ = yaml.load(f, Loader=loader_cls)
    return object_
