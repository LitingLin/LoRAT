import re
from typing import Optional, Sequence, Callable, Protocol


class DataSourceMatcher(Protocol):
    def __call__(self, dataset_name: str, dataset_split: Sequence[str]):
        ...


class DefaultDatasetMatcher:
    def __init__(self,
                 name: Optional[str] = None, name_regex: Optional[str] = None,
                 split: Optional[str] = None):
        self.name = name
        self.name_regex = None
        if name_regex is not None:
            self.name_regex = re.compile(name_regex)
        self.split = split

    def __call__(self, dataset_name: str, dataset_split: Sequence[str]):
        if self.name is not None and self.name != dataset_name:
            return False
        if self.name_regex is not None and self.name_regex.search(dataset_name) is None:
            return False
        if self.split is not None and self.split not in dataset_split:
            return False
        return True


class MultiDataSourceMatcher:
    def __init__(self, matchers: Sequence[Callable[[str, Sequence[str]], bool]]):
        self.matchers = matchers

    def __call__(self, dataset_name: str, dataset_split: Sequence[str]):
        for matcher in self.matchers:
            if matcher(dataset_name, dataset_split):
                return True
        return False
