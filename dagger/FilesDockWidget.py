import os
from PyQt5.QtWidgets import QDockWidget
from PyQt5.QtWidgets import QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QWidget, QAction, QMenu, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

class FilesDockWidget(QDockWidget):

    # Signals
    newFileSelected = pyqtSignal(str)

    def __init__(self, title, parent):
        super().__init__(title, parent)
        
        self.file = None
        self.files = None

        widg = QWidget(self)
        layout = QVBoxLayout(widg)

        self.openSelected = QPushButton('Open Selected', self)
        self.openSelected.clicked.connect(self.handleOpenSelected)

        self.comment = QPushButton('Comment', self)
        self.comment.clicked.connect(self.handleComment)

        self.saveFiles = QPushButton('Save FileList', self)
        self.saveFiles.clicked.connect(self.handleSaveFiles)

        layout.addWidget(self.openSelected)
        layout.addWidget(self.comment)
        layout.addWidget(self.saveFiles)

        #self.metaText = QTextEdit(self)
        #self.metaText.setEnabled(False)
        #self.metaText.setReadOnly(True)
        #self.metaText.setText( "Sample Text\nLine 2" )

        self.filesTable = QTableWidget(self)
        self.filesTable.setEnabled(True)
        self.filesTable.horizontalHeader().setStretchLastSection(True)
        self.filesTable.horizontalHeader().hide()
        self.filesTable.verticalHeader().setDefaultSectionSize( 18 )
        self.filesTable.verticalHeader().hide()
        self.filesTable.setShowGrid(False)
        self.filesTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.filesTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.filesTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.filesTable.cellDoubleClicked[int,int].connect(self.doubleClick)
        self.filesTable.setFocusPolicy(Qt.NoFocus)

        self.filesTable.itemSelectionChanged.connect(self.selectionChanged)

        layout.addWidget(self.filesTable)

        self.setWidget(widg)
        #self.viewMenu.addAction( self.metaDock.toggleViewAction() )

# Buttons: "Open File", "Delete File", "Comment/Uncomment File", "Save FileList", "Open FileList"
# Scroll up/down.
# Show currently open file
# No editing

    def doubleClick( self, row, col ):
        if self.files is not None and row < len(self.files):
            newFile = self.files[row]
            if not newFile.startswith("#"):
                newFile = os.path.abspath(newFile).strip()
                self.newFileSelected.emit(newFile)

    def setFile(self, filename):
        self.file = filename
        with open(self.file,'r') as f:
            lines = f.readlines()
        self.files = list(map(str.strip, lines))
        self.filesTable.setRowCount(len(lines))
        self.filesTable.setColumnCount(1)
        row = 0
        for line in lines:
            self.filesTable.setItem(row,0,QTableWidgetItem(line))
            row += 1
        self.filesTable.selectRow(0)

    def selectNext(self):
        row = self._selectedRow() + 1
        self.filesTable.selectRow(row)
        
    def selectPrev(self):
        row = self._selectedRow() - 1
        self.filesTable.selectRow(row)

    def handleSaveFiles(self):
        filesStr = "\n".join(self.files)
        with open(self.file,'w') as f:
            f.write(filesStr)

    def preferredArea(self):
        return Qt.RightDockWidgetArea

    def _selectedFile(self):
        row = self._selectedRow()
        if row is not None:
            if self.files is not None and row < len(self.files):
                return self.files[row]
        return None

    def _selectedRow(self):
        model = self.filesTable.selectionModel()
        rows = model.selectedRows()
        if len(rows) > 0:
            return rows[0].row()
        return None

    def handleOpenSelected(self):
        row = self._selectedRow()
        if row is not None:
            self.doubleClick(row,0)

    def handleComment(self):
        row = self._selectedRow()
        if row is not None:
            sfile = self._selectedFile()
            if sfile is not None:
                if sfile.startswith("#"):
                    sfile = sfile[1:]
                else:
                    sfile = "#" + sfile
                self.files[row] = sfile
                self.filesTable.setItem(row,0,QTableWidgetItem(sfile))
                self.selectionChanged()

    def selectionChanged( self ):
        sfile = self._selectedFile()
        if sfile is not None:
            if sfile.startswith("#"):
                self.comment.setText("Uncomment")
                self.openSelected.setEnabled(False)
            else:
                self.comment.setText("Comment")
                self.openSelected.setEnabled(True)
            self.comment.setEnabled(True)
        else:
            self.comment.setEnabled(False)
            self.openSelected.setEnabled(False)
