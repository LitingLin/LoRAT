from dataclasses import dataclass
from typing import Iterable, Any, Protocol, Sequence, Optional


class HostDataPipeline:
    def pre_process(self, input_data: Any) -> Any:
        return input_data

    def post_process(self, output_data: Any) -> Any:
        return output_data


@dataclass(frozen=True)
class DataPipeline:
    input: Iterable[Any]
    host: Optional[Sequence[HostDataPipeline]]
