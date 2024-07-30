from typing import Sequence, Mapping
from . import DefaultDatasetMatcher, MultiDataSourceMatcher, DataSourceMatcher


def build_data_source_matcher(rule: Mapping) -> DataSourceMatcher:
    name = None
    name_regex = None
    split = None

    if 'name' in rule:
        name = rule['name']
    if 'name_regex' in rule:
        name_regex = rule['name_regex']
    if 'split' in rule:
        split = rule['split']
    return DefaultDatasetMatcher(name, name_regex, split)


def build_multi_data_source_matcher(rule: Sequence[Mapping]) -> DataSourceMatcher:
    matchers = []
    for matcher_rule in rule:
        matchers.append(build_data_source_matcher(matcher_rule))
    return MultiDataSourceMatcher(matchers)
