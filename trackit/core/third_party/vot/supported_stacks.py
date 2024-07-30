from dataclasses import dataclass


@dataclass
class VOTStack:
    path_name: str


vot_stacks = {
    'vots2024/main': VOTStack('VOTS2023_PATH'),
    'vots2023': VOTStack('VOTS2023_PATH'),
    'tests/multiobject': VOTStack('VOT_TESTS_MULTIOBJECT_PATH'),
}