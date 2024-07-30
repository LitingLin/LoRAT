import yaml
from yaml import CSafeDumper as Dumper


def dump_yaml(object_, path: str, dumper=Dumper):
    with open(path, 'wb') as f:
        yaml.dump(object_, f, encoding='utf-8', default_flow_style=False, Dumper=dumper)
