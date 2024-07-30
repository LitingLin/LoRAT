import multiprocessing
import threading
import queue


def _worker_entry_multiprocessing(task_queue: multiprocessing.Queue):
    while True:
        work = task_queue.get()
        if work == 'quit':
            return
        work[0](*work[1])


def _worker_entry_threading(task_queue: queue.Queue):
    while True:
        work = task_queue.get()
        if work == 'quit':
            return
        work[0](*work[1])
        task_queue.task_done()


class SimpleWorker:
    def __init__(self, max_queue_size: int = 32):
        self._max_queue_size = max_queue_size

    def start(self):
        self.task_queue = queue.Queue(self._max_queue_size)
        self.worker = threading.Thread(target=_worker_entry_threading, args=(self.task_queue,))
        self.worker.start()

    def close(self):
        self.task_queue.put('quit')
        self.worker.join()
        del self.worker
        del self.task_queue

    def submit(self, func, *args):
        self.task_queue.put((func, args))

    def join(self):
        self.task_queue.join()
