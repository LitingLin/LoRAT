from queue import Queue
import threading


class IOThread:
    def start(self):
        self.task_queue = Queue(16)
        self.thread = threading.Thread(target=self._worker_entry)
        self.thread.start()

    def close(self):
        self.task_queue.put('quit')
        self.thread.join()
        del self.thread
        del self.task_queue

    def put(self, func, args=()):
        self.task_queue.put((func, args))

    def join(self):
        self.task_queue.join()

    def _worker_entry(self):
        while True:
            work = self.task_queue.get()
            if work == 'quit':
                return
            work[0](*work[1])
            self.task_queue.task_done()
