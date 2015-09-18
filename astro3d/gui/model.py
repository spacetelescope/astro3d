"""Data Model"""

from __future__ import absolute_import, print_function

from collections import namedtuple
from multiprocessing import (Process, Queue)
from Queue import Empty as QueueEmpty
import threading
from time import sleep

from numpy import concatenate

from ..core.model3d import Model3D
from ..core.meshes import (get_triangles, reflect_mesh)
from ..util.logger import make_logger


__all__ = ['Model']


Task = namedtuple('Task', 'func, args, result')


class Model(object):
    """Data model"""

    image = None

    def __init__(self, signals, logger=None):
        if logger is None:
            logger = make_logger('astro3d model')
        self.logger = logger

        self.signals = signals
        self.signals.NewImage.connect(self.set_image)
        self.signals.ProcessStart.connect(self.thread_process)
        self.signals.ProcessForceQuit.connect(self.process_force_quit)
        self.signals.Quit.connect(self.process_force_quit)
        self.signals.StageChange.connect(self.stagechange)

        self.mesh_thread = None

        self.signals.StageChange('textures', True)
        self.signals.StageChange('intensity', True)
        self.signals.StageChange('spiral_galaxy', True)
        self.signals.StageChange('double_sided', False)

    def stagechange(self, stage, state):
        self.logger.debug('stagechange: stage="{}" state="{}"'.format(stage, state))

        if stage == 'textures':
            self.has_textures = state
        elif stage == 'intensity':
            self.has_intensity = state
        elif stage == 'spiral_galaxy':
            self.spiral_galaxy = state
        elif stage == 'double_sided':
            self.double_sided = state
        self.signals.ModelUpdate()

    def set_image(self, image):
        """Set the image

        Parameters
        ----------
        image: `ginga.AstroImage.AstroImage`
            The image to make the model from.
        """
        self.image = image
        self.signals.ModelUpdate()

    def process(self):
        """Create the 3D model."""
        self.logger.debug('Starting processing...')

        # Setup steps in the thread. Between each step,
        # check to see if stopped.
        m = Model3D(self.image.get_data())

        m.read_all_masks('features/*.fits')

        m.read_stellar_table('features/ngc3344_clusters.txt', 'star_clusters')

        m.has_textures = self.has_textures
        m.has_intensity = self.has_intensity
        m.spiral_galaxy = self.spiral_galaxy
        m.double_sided = self.double_sided

        m.make()

        triset = get_triangles(m.data)
        if m.double_sided:
            triset = concatenate((triset, reflect_mesh(triset)))
        return triset

    def process_force_quit(self, *args, **kwargs):
        """Force quit a process"""
        t = self.mesh_thread
        if t is not None and t.is_alive():
            t.stop()
            t.join()

    def thread_process(self):
        self.process_force_quit()
        self.mesh_thread = MeshThread(args=(self,))
        self.mesh_thread.start()


class MeshThread(threading.Thread):
    """Wait for mesh process data."""

    def __init__(self, **kwargs):
        super(MeshThread, self).__init__(**kwargs)
        self.model = kwargs['args'][0]
        self.logger = self.model.logger
        self._stop = threading.Event()

    @property
    def stopped(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def run(self):
        que = Queue()
        mesh_process = MeshProcess(args=(que, self.model))
        mesh_process.start()

        while not self.stopped:
            try:
                triset = que.get(False)
                self.model.signals.ProcessFinish(triset)
                return
            except QueueEmpty:
                sleep(0.5)
                continue

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
        self.logger.debug('MeshProcess: running...')
        triset = self.model.process()
        self.que.put(triset, False)
