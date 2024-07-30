import importlib
import re


camel_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')

class _BaseFilter:
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __str__(self):
        return f'{self.__class__.__name__}{self.__dict__}'

    def serialize(self):
        if len(self.__dict__) == 0:
            return self.__class__.__name__
        else:
            return self.__class__.__name__, self.__dict__

    @staticmethod
    def deserialize(state):
        if isinstance(state, str):
            class_name = state
            param = None
        elif isinstance(state, (list, tuple)):
            assert len(state) == 2
            assert isinstance(state[0], str)
            assert isinstance(state[1], dict)
            class_name = state[0]
            param = state[1]
        else:
            raise Exception
        paths = class_name.split('_')
        paths = tuple(camel_case_pattern.sub('_', path).lower() for path in paths)
        module_path = 'trackit.datasets.common.filter.' + '.'.join(paths)
        module = importlib.import_module(module_path)
        filter_class = getattr(module, class_name)
        if param is None:
            return filter_class()
        else:
            return filter_class(**param)
