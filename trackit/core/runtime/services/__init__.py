from __future__ import annotations
from typing import Callable, Sequence, Dict, Any, Iterable, Protocol
from dataclasses import dataclass, field


class EventListenerRegister:
    def __init__(self):
        self._listeners = {}
        self._index = 0

    def register(self, listener: Callable, priority: int = 50) -> int:
        handle = self._index
        self._listeners[handle] = (listener, priority)
        self._index += 1
        return handle

    def unregister(self, handle: int) -> None:
        del self._listeners[handle]

    def get(self, handle: int) -> Callable:
        return self._listeners[handle][0]

    def list(self) -> Sequence[Callable]:
        return tuple(listener[0] for listener in sorted(self._listeners.values(), key=lambda x: x[1]))

    def has(self, handle: int) -> bool:
        return handle in self._listeners


class OnStartStopEventCallback(Protocol):
    def __call__(self) -> None:
        ...


class OnEpochBeginEndEventCallback(Protocol):
    def __call__(self, epoch: int, is_train: bool) -> None:
        ...


class OnIterationBeginEndEventCallback(Protocol):
    def __call__(self, is_train: bool) -> None:
        ...


class EpochBasedTrainerEventListenerRegistry:
    def __init__(self):
        self._start_event = EventListenerRegister()
        self._stop_event = EventListenerRegister()
        self._epoch_begin_event = EventListenerRegister()
        self._epoch_end_event = EventListenerRegister()
        self._iteration_begin_event = EventListenerRegister()
        self._iteration_end_event = EventListenerRegister()

    def register_on_start_event_listener(self, listener: OnStartStopEventCallback, priority: int = 50) -> int:
        return self._start_event.register(listener, priority)

    def list_on_start_event_listener(self) -> Iterable[OnStartStopEventCallback]:
        return self._start_event.list()

    def unregister_on_start_event_listener(self, handle: int) -> None:
        self._start_event.unregister(handle)

    def register_on_stop_event_listener(self, listener: OnStartStopEventCallback, priority: int = 50) -> int:
        return self._stop_event.register(listener, priority)

    def list_on_stop_event_listener(self) -> Iterable[OnStartStopEventCallback]:
        return self._stop_event.list()

    def unregister_on_stop_event_listener(self, handle: int) -> None:
        self._stop_event.unregister(handle)

    def register_on_epoch_begin_event_listener(self, listener: OnEpochBeginEndEventCallback, priority: int = 50) -> int:
        return self._epoch_begin_event.register(listener, priority)

    def list_on_epoch_begin_event_listener(self) -> Iterable[OnEpochBeginEndEventCallback]:
        return self._epoch_begin_event.list()

    def unregister_on_epoch_begin_event_listener(self, handle: int):
        self._epoch_begin_event.unregister(handle)

    def register_on_epoch_end_event_listener(self, listener: OnEpochBeginEndEventCallback, priority: int = 50) -> int:
        return self._epoch_end_event.register(listener, priority)

    def list_on_epoch_end_event_listener(self) -> Iterable[OnEpochBeginEndEventCallback]:
        return self._epoch_end_event.list()

    def unregister_on_epoch_end_event_listener(self, handle: int):
        self._epoch_end_event.unregister(handle)

    def register_on_iteration_begin_event_listener(self, listener: OnIterationBeginEndEventCallback, priority: int = 50) -> int:
        return self._iteration_begin_event.register(listener, priority)

    def list_on_iteration_begin_event_listener(self) -> Iterable[OnIterationBeginEndEventCallback]:
        return self._iteration_begin_event.list()

    def unregister_on_iteration_begin_event_listener(self, handle: int):
        self._iteration_begin_event.unregister(handle)

    def register_on_iteration_end_event_listener(self, listener: OnIterationBeginEndEventCallback, priority: int = 50) -> int:
        return self._iteration_end_event.register(listener, priority)

    def list_all_on_iteration_end_event_listener(self) -> Iterable[OnIterationBeginEndEventCallback]:
        return self._iteration_end_event.list()

    def unregister_on_iteration_end_event_listener(self, handle: int):
        self._iteration_end_event.unregister(handle)


class CheckpointStatefulObjectRegistry:
    @dataclass(frozen=True)
    class _Parameters:
        name: str
        get_state_fn: Callable[[], Any]
        set_state_fn: Callable[[Any], None]
        main_process_only: bool

    def __init__(self):
        self.objects = {}

    def register(self, name: str, get_state_fn: Callable[[], Any], set_state_fn: Callable[[Any], None], main_process_only=False, priority: int = 50) -> None:
        assert name not in self.objects
        self.objects[name] = (CheckpointStatefulObjectRegistry._Parameters(name, get_state_fn, set_state_fn, main_process_only), priority)

    def unregister(self, name: str) -> None:
        del self.objects[name]

    def has(self, name: str) -> bool:
        return name in self.objects

    def get(self, name: str) -> CheckpointStatefulObjectRegistry._Parameters:
        return self.objects[name][0]

    def list(self) -> Sequence[CheckpointStatefulObjectRegistry._Parameters]:
        return tuple(v[0] for v in sorted(self.objects.values(), key=lambda x: x[1]))


class BatchCollectiveCommunicationServiceOperatorAllGatherRegistry:
    @dataclass(frozen=True)
    class _Parameter:
        data_prepare_fn: Callable[[], Any]
        on_gathered_fn: Callable[[Sequence[Any]], None]

    def __init__(self):
        self.functions = {}
        self._index = 0

    def register(self, data_prepare_fn: Callable[[], Any], on_gathered_fn: Callable[[Sequence[Any]], None], priority: int=50) -> int:
        handle = self._index
        self.functions[handle] = (BatchCollectiveCommunicationServiceOperatorAllGatherRegistry._Parameter(data_prepare_fn, on_gathered_fn), priority)
        self._index += 1
        return handle

    def has(self, handle: int) -> bool:
        return handle in self.functions

    def unregister(self, handle: int) -> None:
        del self.functions[handle]

    def list(self) -> Sequence[BatchCollectiveCommunicationServiceOperatorAllGatherRegistry._Parameter]:
        sorted_functions = sorted(self.functions.values(), key=lambda x: x[1])
        return tuple(x[0] for x in sorted_functions)

    def get(self, handle: int) -> BatchCollectiveCommunicationServiceOperatorAllGatherRegistry._Parameter:
        return self.functions[handle][0]

    def is_empty(self) -> bool:
        return len(self.functions) == 0


class BatchCollectiveCommunicationServiceOperatorGatherRegistry:
    @dataclass(frozen=True)
    class _Parameter:
        data_prepare_fn: Callable[[], Any]
        on_gathered_fn: Callable[[Sequence[Any]], None]
        dst_rank: int

    def __init__(self):
        self.functions = {}
        self._index = 0

    def register(self, data_prepare_fn: Callable[[], Any], on_gathered_fn: Callable[[Sequence[Any]], None], dst_rank: int = 0, priority: int = 50) -> int:
        handle = self._index
        self.functions[handle] = (BatchCollectiveCommunicationServiceOperatorGatherRegistry._Parameter(data_prepare_fn, on_gathered_fn, dst_rank), priority)
        self._index += 1
        return handle

    def unregister(self, handle: int) -> None:
        del self.functions[handle]

    def get(self, handle: int) -> BatchCollectiveCommunicationServiceOperatorGatherRegistry._Parameter:
        return self.functions[handle][0]

    def has(self, handle: int) -> bool:
        return handle in self.functions

    def list(self) -> Sequence[BatchCollectiveCommunicationServiceOperatorGatherRegistry._Parameter]:
        sorted_functions = sorted(self.functions.values(), key=lambda x: x[1])
        return tuple(x[0] for x in sorted_functions)

    def is_empty(self) -> bool:
        return len(self.functions) == 0


@dataclass()
class BatchCollectiveCommunicationServiceRegistry:
    all_gather: BatchCollectiveCommunicationServiceOperatorAllGatherRegistry = field(default_factory=BatchCollectiveCommunicationServiceOperatorAllGatherRegistry)
    gather: BatchCollectiveCommunicationServiceOperatorGatherRegistry = field(default_factory=BatchCollectiveCommunicationServiceOperatorGatherRegistry)

    def is_empty(self):
        return self.all_gather.is_empty() and self.gather.is_empty()


@dataclass()
class ServicesRegistry:
    checkpoint: CheckpointStatefulObjectRegistry = field(default_factory=CheckpointStatefulObjectRegistry)
    event: EpochBasedTrainerEventListenerRegistry = field(default_factory=EpochBasedTrainerEventListenerRegistry)
    batch_collective_communication: BatchCollectiveCommunicationServiceRegistry = field(default_factory=BatchCollectiveCommunicationServiceRegistry)
