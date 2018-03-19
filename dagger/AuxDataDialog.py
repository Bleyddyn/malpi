#from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, QMenu, QMessageBox, QApplication, QDesktopWidget, qApp, QPushButton
from PyQt5.QtWidgets import QGridLayout
#from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
#from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QSlider, QListView, QTreeView, QAbstractItemView, QComboBox, QFrame
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QComboBox, QDialogButtonBox
#from PyQt5.QtWidgets import QDialog, QFileDialog, QDockWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QDoubleValidator, QIntValidator
#from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject

# dialog box with name, type (categorical, continuous), count/range inputs

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

        catCountLabel = QLabel("Number of Categories")
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
        grid.addWidget(catCountLabel, row, 0)
        grid.addWidget(self.catCount, row, 1)

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

        dtype = self.dataType.currentText()
        auxMeta = {"name":auxName, "type":dtype}
        if dtype == "Categorical":
            try:
                catCount = int(self.catCount.text())
            except:
                catCount = None
            if catCount is None or catCount < 0:
                self.warningLabel.setText("Number of categories must be a number greater than zero")
                return
            auxMeta["count"] = catCount
            auxMeta["default"] = 0
        elif dtype == "Continuous":
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
        self.rangeMin.setEnabled(False)
        self.rangeMax.setEnabled(False)

        if newValue == "Categorical":
            self.catCount.setEnabled(True)
        elif newValue == "Continuous":
            self.rangeMin.setEnabled(True)
            self.rangeMax.setEnabled(True)

    def getMeta(self):
        return self.meta
