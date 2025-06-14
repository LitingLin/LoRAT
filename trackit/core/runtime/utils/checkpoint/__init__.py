from typing import Protocol, Any


class model_state_save_fn(Protocol):
    def __call__(self, checkpoint_path: str, exclude_frozen_parameters: bool) -> str:
        ...

class application_state_save_fn(Protocol):
    def __call__(self, checkpoint_path: str) -> Any:
        ...

class application_state_load_fn(Protocol):
    def __call__(self, state: Any, checkpoint_path: str) -> None:
        ...
