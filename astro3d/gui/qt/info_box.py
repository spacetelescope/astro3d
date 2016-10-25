"""Info Box"""
from qtpy import QtWidgets

__all__ = ['InfoBox']


class InfoBox(QtWidgets.QMessageBox):

    def show_error(self, label, error):
        """ Show error

        Parameters
        ----------
        label: str
            The label of the message.

        error: str
            Detailed error message
        """
        self.setText(str(label))
        self.setInformativeText(str(error))
        self.exec_()
