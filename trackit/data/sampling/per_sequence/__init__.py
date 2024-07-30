from typing import Protocol, Tuple


class RandomAccessiblePerSequenceSampler(Protocol):
    def __getitem__(self, index: int) -> Tuple[int, int]:
        ...

    def __len__(self) -> int:
        ...
