from dataclasses import dataclass


@dataclass
class VOTStack:
    path_name: str


vot_stacks = {
    'vot2020/shortterm': VOTStack('VOT2020ST_PATH'),
    'vot2022/shorttermbox': VOTStack('VOT2022STB_PATH'),
    'vots2025/main': VOTStack('VOTS2023_PATH'),
    'vots2024/main': VOTStack('VOTS2023_PATH'),
    'vots2023': VOTStack('VOTS2023_PATH'),
    'tests/multiobject': VOTStack('VOT_TESTS_MULTIOBJECT_PATH'),
}