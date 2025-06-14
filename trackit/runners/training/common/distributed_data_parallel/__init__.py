from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedDataParallelOption:
    find_unused_parameters: bool
    gradient_as_bucket_view: bool
    static_graph: bool
    convert_sync_batchnorm: bool
    model_params_and_buffers_to_ignore: Optional[Tuple[str, ...]] = None
    criterion_params_and_buffers_to_ignore: Optional[Tuple[str, ...]] = None
