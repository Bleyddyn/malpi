import sys

from PyQt5.QtWidgets import QLabel, QComboBox, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject

class AuxDataUI(QObject):

    # Signals
    auxDataChanged = pyqtSignal(str)

    @staticmethod
    def create(auxMeta, dataContainer, count):
        """ Takes a dictionary of auxiliary data attributes and returns an appropriate object
            for making and handling the UI.
        """
        atype = auxMeta["type"]
        for subclass in AuxDataUI.__subclasses__():
            if subclass.handlesType(atype):
                return subclass(auxMeta, dataContainer, count)
        raise ValueError( "Unknown auxiliary data type: {}".format( atype ) )

    @classmethod
    def handlesType(cls, atype):
        return False

class AuxDataCatUI(AuxDataUI):
    """ A class that takes a dictionary of auxiliary data attributes then
        makes and manages a set of UI elements to allow input/output of the data.
    """

    @classmethod
    def handlesType(cls, atype):
        if "categorical" == atype:
            return True
        return False

    def __init__(self, auxMeta, dataContainer, count):
        super().__init__()

        self.meta = auxMeta
        self.data = dataContainer
        self.data.addAuxData(auxMeta)
        self.keys = [str(idx+1) for idx, val in enumerate(auxMeta["categories"])] # keybindings
        self.index = 0
        self.include_none = True
        self.initUI(count)

    def initUI(self, count):
        self.nameLabel = QLabel(self.meta["name"])
        self.dataLabels = []
        for i in range(count):
            cb = self._makeComboBox()
            self.dataLabels.append( cb )

    def getUI(self):
        return (self.nameLabel, self.dataLabels)

    def getMeta(self):
        return self.meta

    def update(self, index):
        self.index = index
        for i in range( len(self.dataLabels) ):
            il = index - 2 + i
            if il >= 0 and il < self.data.count():
                data = self.data.auxDataAtIndex( self.meta["name"], il )
                if data is None:
                    data = ""
                self.dataLabels[i].setCurrentText( self.data.auxDataAtIndex( self.meta["name"], il ) )
                self.dataLabels[i].setEnabled(True)
            else:
                self.dataLabels[i].setCurrentText( "" )
                self.dataLabels[i].setEnabled(False)

    def handleKeyPressEvent(self, e, index):
        if e.text() in self.keys:
            try:
                idx = int(e.text()) - 1
            except ValueError:
                return False
            if idx >= 0 and idx < len(self.meta["categories"]):
                self.changeCurrentAction( index, self.meta["categories"][idx] )
                return True
        return False

    def changeCurrentAction(self, index, newCat, label_index=2):
        self.dataLabels[label_index].setCurrentText( newCat )
        self.data.setAuxDataAtIndex( self.meta["name"], newCat, index )
        self.auxDataChanged.emit(self.meta["name"])

    def _makeComboBox(self):
        ae = QComboBox()
        if self.include_none:
            ae.addItem("")
        for aclab in self.meta["categories"]:
            ae.addItem(aclab)
        ae.setInsertPolicy(QComboBox.NoInsert)
        ae.activated[str].connect(self._actionEdited)
        return ae

    def _actionEdited(self, newValue):
        idx = self.dataLabels.index(self.sender())
        if "" == newValue:
            newValue = None
        self.data.setAuxDataAtIndex(self.meta["name"], newValue, self.index + -2 +idx )
        self.auxDataChanged.emit(self.meta["name"])

if __name__ == "__main__":
    auxMeta = { "name": "TestAux", "type": "categorical", "categories": ["cat1", "cat2", "cat3"]}
    app = QApplication(sys.argv)
    obj = AuxDataUI.create(auxMeta, {}, 5)
    #obj.exec()
    sys.exit(app.exec_())

