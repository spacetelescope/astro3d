from multiprocessing import (Process, Queue)
from Queue import Empty as QueueEmpty
import threading
from time import sleep

from ..gui import signaldb


class MeshThread(threading.Thread):
    """Wait for mesh process data."""

    def __init__(self, **kwargs):
        super(MeshThread, self).__init__(**kwargs)
        self.controller = kwargs['args'][0]
        self.logger = self.controller.logger
        self._stop = threading.Event()

    @property
    def stopped(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def run(self):
        que = Queue()
        mesh_process = MeshProcess(args=(que, self.controller.model))
        mesh_process.daemon = True
        mesh_process.start()

        while mesh_process.is_alive() and not self.stopped:
            try:
                triset, model3d = que.get(False)
                signaldb.ProcessFinish(triset, model3d)
            except QueueEmpty:
                sleep(0.5)

        # Terminate process
        que.cancel_join_thread()
        que.close()
        mesh_process.terminate()


class MeshProcess(Process):
    """The mesh thread"""

    def __init__(self, **kwargs):
        super(MeshProcess, self).__init__(**kwargs)
        self.que = kwargs['args'][0]
        self.model = kwargs['args'][1]
        self.logger = self.model.logger

    def run(self):
        try:
            triset = self.model.process()
            self.que.put(triset, False)
        finally:
            self.que.close()
