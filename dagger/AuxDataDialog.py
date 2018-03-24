import sys

#from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, QMenu, QMessageBox, QApplication, QDesktopWidget, qApp, QPushButton
from PyQt5.QtWidgets import QGridLayout, QApplication
#from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
#from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QSlider, QListView, QTreeView, QAbstractItemView, QComboBox, QFrame
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QComboBox, QDialogButtonBox
#from PyQt5.QtWidgets import QDialog, QFileDialog, QDockWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QDoubleValidator, QIntValidator
#from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject

# dialog box with name, type (categorical, continuous), count/range inputs

class DaggerComboBox(QComboBox):
    def keyPressEvent(self,e):
        super(DaggerComboBox,self).keyPressEvent(e)

        if e.key() == Qt.Key_Enter or e.key() == Qt.Key_Return:
# accept enter/return events so they won't be ever propagated to the parent dialog..
            e.accept()

class AuxDataDialog(QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        self.meta = None

        nameLabel = QLabel("Aux Data Name")
        self.nameEdit = QLineEdit()

        typeLabel = QLabel("Aux Data Type")
        self.dataType = QComboBox()
        self.dataType.addItem("Categorical")
        self.dataType.addItem("Continuous")
        self.dataType.setInsertPolicy(QComboBox.NoInsert)
        self.dataType.activated[str].connect(self.typeEdited)

        catCountLabel = QLabel("Category Labels")
        self.catCount = QLineEdit()
        self.catCount.setEnabled(True)
        val = QIntValidator(self)
        val.setBottom(1)
        self.catCount.setValidator( val )

        rangeLabel = QLabel("Data Range")
        self.rangeMin = QLineEdit()
        self.rangeMax = QLineEdit()
        self.rangeMin.setEnabled(False)
        self.rangeMax.setEnabled(False)
        self.rangeMin.setValidator( QDoubleValidator(self) )
        self.rangeMax.setValidator( QDoubleValidator(self) )

        # TODO: Add a field for default value

        self.okButton = QPushButton("Create")
        self.okButton.clicked.connect(self.handleOkButton)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.handleCancelButton)

        self.warningLabel = QLabel()

        grid = QGridLayout()
        grid.setSpacing(10)

        row = 0
        grid.addWidget(nameLabel, row, 0)
        grid.addWidget(self.nameEdit, row, 1)

        row += 1
        grid.addWidget(typeLabel, row, 0)
        grid.addWidget(self.dataType, row, 1)

        row += 1
        #self.categories = QComboBox(self)
        self.categories = DaggerComboBox(self)
        self.categories.setInsertPolicy(QComboBox.InsertAtBottom)
        self.categories.setEditable(True)
        #self.categories.addItem("Category1")
        grid.addWidget(catCountLabel, row, 0)
        grid.addWidget(self.categories, row, 1)

        row += 1
        grid.addWidget(rangeLabel, row, 0)
        grid.addWidget(self.rangeMin, row, 1)
        grid.addWidget(self.rangeMax, row, 2)

        row += 1
        grid.addWidget(self.warningLabel, row, 0, 1, 3)

        row += 1
        buttonBox = QDialogButtonBox(Qt.Horizontal)
        buttonBox.addButton(self.okButton, QDialogButtonBox.AcceptRole)
        buttonBox.addButton(self.cancelButton, QDialogButtonBox.RejectRole)
        #buttonBox.accepted.connect(self.accept)
        #buttonBox.rejected.connect(self.reject)

        grid.addWidget(buttonBox, row, 1, 1, 2)

        self.setLayout(grid)

        self.setWindowTitle("Add Auxiliary Data")


    def handleOkButton(self):
        auxName = self.nameEdit.text()
        if len(auxName) == 0:
            self.warningLabel.setText("You must enter a name for this Auxiliary Data Type")
            return

        dtype = self.dataType.currentText().lower()
        auxMeta = {"name":auxName, "type":dtype}
        if dtype == "categorical":
            auxMeta["default"] = 0
            cats = []
            for idx in range(self.categories.count()):
                cats.append(self.categories.itemText(idx))
            auxMeta["categories"] = cats
        elif dtype == "continuous":
            try:
                rmin = float(self.rangeMin.text())
                rmax = float(self.rangeMax.text())
            except:
                rmin = None
                rmax = None
            if rmin is not None and rmax is not None:
                if rmin >= rmax:
                    self.warningLabel.setText("Maximum must be greater than minimum")
                    return
            auxMeta["min" ] = rmin
            auxMeta["max" ] = rmax
            auxMeta["default"] = rmin
        else:
            self.warningLabel.setText("Invalid data type")
            return

        self.meta = auxMeta

        self.accept()

    def handleCancelButton(self):
        self.meta = None
        self.reject()

    def typeEdited(self, newValue):
        self.catCount.setEnabled(False)
        self.categories.setEnabled(False)
        self.rangeMin.setEnabled(False)
        self.rangeMax.setEnabled(False)

        if newValue.lower() == "categorical":
            self.categories.setEnabled(True)
        elif newValue.lower() == "continuous":
            self.rangeMin.setEnabled(True)
            self.rangeMax.setEnabled(True)

    def getMeta(self):
        return self.meta

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #sys.exit(app.exec_())
    dlg = AuxDataDialog(None)
    nMode = dlg.exec()
    if nMode == QDialog.Accepted:
        print( "Meta: {}".format( dlg.getMeta() ) )
    else:
        print( "Cancelled, don't create auxiliary data" )
