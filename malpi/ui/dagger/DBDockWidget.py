""" A DockWidget for displaying a list of Sources in a database
      and letting the user select one to open in the main window.
"""

import os
import sqlite3

from PyQt5.QtWidgets import QDockWidget
from PyQt5.QtWidgets import QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QWidget, QAction, QMenu, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog, QFileDialog

from malpi.ui.SqliteFormat import SqliteFormat

class DBDockWidget(QDockWidget):

    # Signals
    newFileSelected = pyqtSignal(str)

    def __init__(self, title, parent):
        super().__init__(title, parent)
        
        self.filename = None
        self.connection = None
        self.sources = None

        widg = QWidget(self)
        layout = QVBoxLayout(widg)

        self.openDatabase = QPushButton('Open Database', self)
        self.openDatabase.clicked.connect(self.handleOpenDatabase)

        self.openSelected = QPushButton('Open Selected', self)
        self.openSelected.clicked.connect(self.handleOpenSelected)

        self.saveFiles = QPushButton('Save FileList', self)
        self.saveFiles.clicked.connect(self.handleSaveFiles)

        layout.addWidget(self.openDatabase)
        layout.addWidget(self.openSelected)
        layout.addWidget(self.saveFiles)

        #self.metaText = QTextEdit(self)
        #self.metaText.setEnabled(False)
        #self.metaText.setReadOnly(True)
        #self.metaText.setText( "Sample Text\nLine 2" )

        self.SourcesTable = QTableWidget(self)
        self.SourcesTable.setEnabled(True)
        self.SourcesTable.horizontalHeader().setStretchLastSection(True)
        self.SourcesTable.horizontalHeader().hide()
        self.SourcesTable.verticalHeader().setDefaultSectionSize( 18 )
        self.SourcesTable.verticalHeader().hide()
        self.SourcesTable.setShowGrid(False)
        self.SourcesTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.SourcesTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.SourcesTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.SourcesTable.cellDoubleClicked[int,int].connect(self.doubleClick)
        self.SourcesTable.setFocusPolicy(Qt.NoFocus)

        self.SourcesTable.itemSelectionChanged.connect(self.selectionChanged)

        layout.addWidget(self.SourcesTable)

        self.setWidget(widg)
        #self.viewMenu.addAction( self.metaDock.toggleViewAction() )

    def handleOpenDatabase(self):
        """ Show a file open dialog for database files, ending in .db or .sqlite.
        Create a connection object if the user selects a file."""
        od = QFileDialog(self, 'Open a database containing DonkeyCar data')
        od.setAcceptMode(QFileDialog.AcceptOpen)
        od.setFileMode(QFileDialog.ExistingFile)
        od.setOption(QFileDialog.DontUseNativeDialog, True);
        od.setNameFilter("Database files (*.db *.sqlite)")

        nMode = od.exec()
        if nMode == QDialog.Accepted:
            _fnames = od.selectedFiles() # QStringList 
             
            try:
                if 1 == len(_fnames):
                    self.setDatabase( _fnames[0] )
            except Exception as ex:
                msg = "Error opening a database: {}".format( str(ex) )
                print( msg )

    def setDatabase( self, filename ):
        self.filename = filename
        self.connection = sqlite3.connect(self.filename)
        self.connection.row_factory = sqlite3.Row # Allows access as a dictionary, with column names as keys
        self.getSources()
        SqliteFormat.setConnection( self.connection )

    def getSources( self ):
        sql = "Select source_id, name, full_path from Sources;"
        cursor = self.connection.cursor()
        self.sources = cursor.execute(sql).fetchall()
        self.fillSourcesTable()

    def fillSourcesTable(self):
        self.SourcesTable.setRowCount(len(self.sources))
        self.SourcesTable.setColumnCount(1)
        row = 0
        for source in self.sources:
            self.SourcesTable.setItem(row,0,QTableWidgetItem(source[1]))
            row += 1
        self.SourcesTable.selectRow(0)

    def doubleClick( self, row, col ):
        if self.sources is not None and row < len(self.sources):
            self.newFileSelected.emit( SqliteFormat.source_tag + str(self.sources[row][0]) )

    def selectNext(self):
        row = self._selectedRow()
        if row is not None:
            row = row + 1
            self.SourcesTable.selectRow(row)
        
    def selectPrev(self):
        row = self._selectedRow()
        if row is not None:
            row = row - 1
            self.SourcesTable.selectRow(row)

    def handleSaveFiles(self):
        filesStr = "\n".join(self.sources)
        with open(self.file,'w') as f:
            f.write(filesStr)

    def preferredArea(self):
        return Qt.RightDockWidgetArea

    def _selectedFile(self):
        row = self._selectedRow()
        if row is not None:
            if self.sources is not None and row < len(self.sources):
                return self.sources[row]
        return None

    def _selectedRow(self):
        model = self.SourcesTable.selectionModel()
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
                self.sources[row] = sfile
                self.SourcesTable.setItem(row,0,QTableWidgetItem(sfile))
                self.selectionChanged()

    def selectionChanged( self ):
        sfile = self._selectedFile()
        if sfile is not None:
            self.openSelected.setEnabled(True)
        else:
            self.openSelected.setEnabled(False)
