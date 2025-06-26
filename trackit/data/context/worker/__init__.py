import os.path
from typing import Optional, Tuple
from dataclasses import dataclass, field
from trackit.core.runtime.context.epoch import get_current_epoch_context


@dataclass()
class _MutableVariables:
    output_path_exists: bool = False


@dataclass(frozen=True)
class WorkerInfo:
    worker_id: int
    num_local_workers: int
    rank: int
    world_size: int
    epoch: int
    is_train: int
    rng_seed: Tuple[int, ...]
    in_background_process: bool
    _output_path: str
    __mutable_variables: _MutableVariables = field(default_factory=_MutableVariables)

    def get_output_path(self, create_if_not_exists: bool = True) -> Optional[str]:
        output_path = self._output_path
        if output_path is not None:
            output_path = os.path.join(output_path, f'worker_rank_{self.rank}_id_{self.worker_id}')
        if create_if_not_exists:
            if not self.__mutable_variables.output_path_exists and output_path is not None:
                os.makedirs(output_path, exist_ok=True)
                self.__mutable_variables.output_path_exists = True
        return output_path

    def get_global_worker_id(self):
        return self.rank * self.num_local_workers + self.worker_id


__worker_info: Optional[WorkerInfo] = None


# return None if current process is not a worker
def get_current_worker_info() -> Optional[WorkerInfo]:
    return __worker_info


def _set_worker_info(worker_id: int, num_local_workers: int, rank: int, world_size: int, epoch: int, is_train: bool,
                     seed: int, in_background_process: bool, output_path: Optional[str]):
    rng_seed = (seed, rank, epoch, int(is_train))
    global __worker_info
    __worker_info = WorkerInfo(worker_id, num_local_workers, rank, world_size, epoch, is_train, rng_seed,
                               in_background_process, output_path)


def _unset_worker_info():
    global __worker_info
    __worker_info = None


class WorkerInfoInitializer:
    def __init__(self, rank: int, seed: int, num_local_workers: int, world_size: int, in_background_process: bool):
        assert num_local_workers > 0
        self._epoch = 0
        self._rank = rank
        self._seed = seed
        self._num_local_workers = num_local_workers
        self._world_size = world_size
        self._is_train = False
        self._in_background_process = in_background_process

    def on_epoch_begin(self, epoch: int, is_train: bool):
        self._epoch = epoch
        self._is_train = is_train
        self._output_path = get_current_epoch_context().get_current_epoch_output_path(create_if_not_exists=False)
        if not self._in_background_process:
            _set_worker_info(0, self._num_local_workers, self._rank, self._world_size, self._epoch,
                             self._is_train, self._seed, self._in_background_process, self._output_path)

    def worker_init_fn(self, worker_id: int):
        _set_worker_info(worker_id, self._num_local_workers, self._rank, self._world_size, self._epoch,
                         self._is_train, self._seed, self._in_background_process, self._output_path)

    def on_epoch_end(self, epoch: int, is_train: bool):
        assert self._epoch == epoch and self._is_train == is_train
        del self._output_path
        if not self._in_background_process:
            _unset_worker_info()
