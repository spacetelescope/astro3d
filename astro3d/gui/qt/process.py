"""Thread to process 3D model"""
import traceback

from astropy import log
from numpy import concatenate
from qtpy import QtCore
from qtpy.QtCore import Signal as pyqtSignal

from ...core.meshes import (make_triangles, reflect_triangles)

from ...gui import signaldb

__all__ = ['MeshThread']

# Configure logging
log.setLevel('DEBUG')


class MeshWorker(QtCore.QObject):
    """The mesh thread

    Parameters
    ----------
    model: astro3d.Model3d
        The model to process.

    make_params: dict
        Other parameters to create the model

    Signals
    -------
    finished(results): pyqtSignal emitted
        Emitted when processing is successfully
        completed. The results is a tuple with
            triset: 3D mesh for display
            model3d: astro3d.core.Model3d of the
                     computed model.

    aborted: pyqtSignal emitted
        Emitted when the processing has been aborted.
        Usually this is a result of another thread
        emitting the abort signal.

    exception: pyqtSignal emitted
        If an exception or other error condition
        occurs, this will be emitted with an
        Exception as argument

    abort: pyqtSignal received
        If received, processing will stop and the
        aborted signal will be emitted

    """
    abort = pyqtSignal()
    aborted = pyqtSignal()
    finished = pyqtSignal(tuple)
    exception = pyqtSignal(Exception)

    def __init__(self, model3d, make_params):
        super(MeshWorker, self).__init__()
        self.model3d = model3d
        self.make_params = make_params
        self._aborted = False
        self.abort.connect(self._abort)

    def run(self):
        try:
            self.model3d.make(**self.make_params)
            triset = make_triangles(self.model3d.data)

            if self.make_params['double_sided']:
                triset = concatenate((triset, reflect_triangles(triset)))
        except Exception as e:
            log.debug(traceback.format_exc())
            self.exception.emit(e)
            return

        if not self._aborted:
            self.finished.emit((triset, self.model3d))

    def _abort(self):
        self._aborted = True
        self.aborted.emit()


class MeshThread(object):
    def __init__(self, model3d, make_params):
        mesh_worker = MeshWorker(model3d, make_params)
        self.mesh_worker = mesh_worker
        worker_thread = QtCore.QThread()
        self.worker_thread = worker_thread
        mesh_worker.moveToThread(worker_thread)

        worker_thread.started.connect(mesh_worker.run)
        worker_thread.finished.connect(self.cleanup)

        mesh_worker.finished.connect(self.finished)
        mesh_worker.finished.connect(
            lambda x: worker_thread.quit()
        )
        mesh_worker.aborted.connect(
            lambda: self.mesh_worker_fail('Processing aborted.')
        )
        mesh_worker.exception.connect(
            lambda e: self.mesh_worker_fail('Processing error', e)
        )
        signaldb.ProcessForceQuit.connect(
            self.mesh_worker.abort.emit,
            single_shot=True
        )

        worker_thread.start()

    def finished(self, results):
        triset, model3d = results
        signaldb.ProcessFinish(triset, model3d)

    def cleanup(self):
        signaldb.ProcessForceQuit.clear(single_shot=True)
        self.worker_thread.deleteLater()
        self.mesh_worker.deleteLater()

    def mesh_worker_fail(self, message='', error_text=''):
        self.worker_thread.quit()
        signaldb.ProcessFail(message, error_text)
