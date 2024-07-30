import numpy as np
from typing import Optional, Tuple


class DynamicTaskScheduler:
    def __init__(self, tasks: np.ndarray, world_size: int):
        '''
        :param tasks: shape: (num_tasks,), containing the number of steps of each task
        :param world_size: number of ranks
        '''
        assert np.all(tasks > 0)
        self._tasks = tasks
        self._world_size = world_size
        # self.reset()

    def reset(self):
        self._exhausted_index = 0
        self._running_tasks = [{} for _ in range(self._world_size)]

    def get_next_batch(self, rank_id: int, rank_iteration: int, batch_size: int) -> Optional[Tuple[Tuple[int, int], ...]]:
        # torch dataloader is order preserved
        assert rank_id < self._world_size

        rank_running_tasks = self._running_tasks[rank_id]
        batch = []

        for task_index in list(rank_running_tasks.keys()):
            sequence_last_iteration, step_index = rank_running_tasks[task_index]
            if rank_iteration <= sequence_last_iteration:
                continue

            batch.append((task_index, step_index))

            # look forward
            step_index += 1
            if step_index == self._tasks[task_index]:
                del rank_running_tasks[task_index]
            else:
                rank_running_tasks[task_index] = (rank_iteration, step_index)

            if len(batch) == batch_size:
                break

        while len(batch) != batch_size:
            if self._exhausted_index == len(self._tasks):
                break

            batch.append((self._exhausted_index, 0))

            if self._tasks[self._exhausted_index] > 1:
                rank_running_tasks[self._exhausted_index] = (rank_iteration, 1)

            self._exhausted_index += 1

        if len(batch) == 0:
            return None
        return tuple(batch)

    def get_next(self, rank_id: int, rank_iteration: int) -> Optional[Tuple[int, int]]:
        batch = self.get_next_batch(rank_id, rank_iteration, 1)
        if len(batch) == 0:
            return None
        else:
            return batch[0]

    def is_done(self):
        return self._exhausted_index == len(self._tasks) and all(len(per_rank_running_task) == 0 for per_rank_running_task in self._running_tasks)
