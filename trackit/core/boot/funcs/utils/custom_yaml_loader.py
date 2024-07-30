import yaml
from yaml import CSafeLoader as Loader
from trackit.core.runtime.global_constant import get_global_constant
import os
import itertools


class CustomLoader(Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]

        super(CustomLoader, self).__init__(stream)

    def include(self, node):
        if not isinstance(node, yaml.ScalarNode):
            raise TypeError('!include tag must be a scalar')
        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, CustomLoader)

    def const(self, node):
        if isinstance(node, yaml.ScalarNode):
            const_name = self.construct_scalar(node)
            return get_global_constant(const_name)
        elif isinstance(node, yaml.SequenceNode):
            const_name = self.construct_sequence(node)
            return get_global_constant(*const_name)
        else:
            raise TypeError('!env tag must be a scalar or sequence')

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


CustomLoader.add_constructor('!include', CustomLoader.include)
CustomLoader.add_constructor('!const', CustomLoader.const)
CustomLoader.add_constructor('!concat', CustomLoader.concat)


def load_yaml(path: str, loader=CustomLoader):
    with open(path, 'rb') as f:
        object_ = yaml.load(f, Loader=loader)
    return object_
