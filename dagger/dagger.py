#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" A tool for modifying data collected by MaLPi.
An algorithm called DAgger [need ref] involves collecting data while an agent/robot is running its policy,
then having an 'expert' go over the data and correct any mistakes the agent made,
then retraining on the cleaned up data.

TODO:
1) Add in-app help (keyboard shortcuts, etc)
3) Make it obvious which action will be changed via keyboard shortcut
4) Pre-generate QImages and cache them
5) Load/Save DonkeyCar tub files
"""

import sys
import os
import argparse

from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, QMenu, QMessageBox, QApplication, QDesktopWidget, qApp, QPushButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QSlider, QListView, QTreeView, QAbstractItemView, QComboBox, QFrame
from PyQt5.QtWidgets import QDialog, QFileDialog, QDockWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject

from DriveFormat import DriveFormat
import MalpiFormat
import TubFormat

class Communicate(QObject):
    closeApp = pyqtSignal() 

class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.data = None
        self.index = 0
        self.gridWidth = 5
        self.viewMenu = None
        self.metaDock = None
        self.statsDock = None
        self.path = ""

        self.initUI()

    def initUI(self):               

        wid = QWidget(self)
        self.setCentralWidget(wid)

        self.c = Communicate()
        self.c.closeApp.connect(self.close)       
        
        #self.setMouseTracking(True)

        self.updateWindowTitle()
        self.statusBar().showMessage('Ready')
        self.initMenus()

        self.setGeometry(200, 200, 1200, 800)
        self.centerWindowOnScreen()

        self.show()
        
    def initMenus(self):
        #exitAct = QAction(QIcon('exit.png'), 'Exit', self)        
        #exitAct = QAction(QIcon('exit24.png'), 'Exit', self)
        exitAct = QAction('Exit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        openFile = QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.triggered.connect(self.openDialog)

        saveFile = QAction('Save', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.triggered.connect(self.saveData)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu.addAction(exitAct)

        viewMenu = menubar.addMenu('View')
        
        viewStatAct = QAction('Statusbar', self, checkable=True)
        viewStatAct.setStatusTip('Statusbar')
        viewStatAct.setChecked(True)
        viewStatAct.triggered.connect(self.toggleMenu)
        
        viewMenu.addAction(viewStatAct)

        self.viewMenu = viewMenu

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAct)
        self.toolbar.addAction(openFile)

    def makeActionComboBox(self):
        ae = None

        ot = self.data.outputTypes()
        if ot[0]["type"] == "categorical":
            ae = QComboBox(self)
            for aclab in ot[0]["categories"]:
                ae.addItem(aclab)
            ae.setInsertPolicy(QComboBox.NoInsert)
            ae.activated[str].connect(self.actionEdited)
        elif ot[0]["type"] == "continuous":
            ae = QLineEdit(self)
            ae.setEnabled(False)

        if ae is None:
            ae = QLineEdit(self)
        return ae

    def initGrid(self):
        if self.data is None:
            return

        # This needs to be generalized to handle multiple input and output types
        self.imageLabels = []
        self.actionLabels = []
        for i in range(self.gridWidth):
            self.imageLabels.append( QLabel(self) )
            cb = self.makeActionComboBox()
            self.actionLabels.append( cb )


        grid = QGridLayout()
        grid.setSpacing(10)

        otypes = self.data.outputTypes()
        itypes = self.data.inputTypes()

        row = 1
        grid.addWidget(QLabel(itypes[0]["name"]), row, 0)
        for i in range(len(self.imageLabels)):
            grid.addWidget(self.imageLabels[i], row, i+1)

        row += 1
        grid.addWidget(QLabel(otypes[0]["name"]), row, 0)
        for i in range(len(self.actionLabels)):
#            if i == 2: # should be gridWidth / 2
#                cbframe = QFrame(self)
#                self.actionLabels[i].setParent( cbframe )
#                cbframe.setStyleSheet("QFrame { border: 2px solid black; }")
#                grid.addWidget(cbframe, row, i+1)
#            else:
            grid.addWidget(self.actionLabels[i], row, i+1)

        row += 1
        self.indexes = []
        for i in range(self.gridWidth):
            idx = QLabel(self)
            idx.setAlignment( Qt.AlignCenter )
            self.indexes.append( idx )
        for i in range(len(self.indexes)):
            grid.addWidget(self.indexes[i], row, i+1)

        row += 1
        sld = QSlider(Qt.Horizontal, self)
        self.slider = sld
        sld.setFocusPolicy(Qt.NoFocus)
        sld.valueChanged[int].connect(self.changeValue)
        grid.addWidget(sld, row, 0, 1, self.gridWidth+1)

        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(grid)

        if self.metaDock is None:
            self.metaDock = QDockWidget("Drive Info", self)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.metaDock)
            self.metaText = QTextEdit(self)
            self.metaText.setEnabled(False)
            self.metaText.setReadOnly(True)
            #self.metaText.setMinimumWidth( 200.0 )
            self.metaDock.setWidget(self.metaText)
            self.viewMenu.addAction( self.metaDock.toggleViewAction() )
        self.metaText.setText( self.data.meta )

        if self.statsDock is None:
            self.statsDock = QDockWidget("Action Stats", self)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.statsDock)
            self.statsTable = QTableWidget(self)
            self.statsTable.setEnabled(False)
            self.statsTable.horizontalHeader().hide()
            self.statsTable.verticalHeader().setDefaultSectionSize( 18 )
            self.statsTable.verticalHeader().hide()
            self.statsTable.setShowGrid(False)
            self.statsDock.setWidget(self.statsTable)
            self.viewMenu.addAction( self.statsDock.toggleViewAction() )
        self.updateStats()

    def updateWindowTitle(self):
        path = ""
        edit_msg = ""
        if len(self.path) > 0:
            path = ": " + os.path.basename(self.path)
        if self.data is not None:
            if not self.data.isClean():
                edit_msg = " --Edited"
        self.setWindowTitle('DAgger Tool' + path + edit_msg)

    def changeValue(self, value):
        self.index = value
        self.updateImages()

    def actionEdited(self, newValue):
        idx = self.actionLabels.index(self.sender())
        self.data.setActionForIndex( newValue, self.index + idx )
        self.updateWindowTitle()
        self.updateStats()

    def toggleMenu(self, state):
        if state:
            self.statusBar().show()
        else:
            self.statusBar().hide()

    def openDialog(self):
        od = QFileDialog(self, 'Open MaLPi drive data directory')
        od.setAcceptMode(QFileDialog.AcceptOpen)
        od.setFileMode(QFileDialog.Directory);
        od.setOption(QFileDialog.DontUseNativeDialog, True);

        # Try to select multiple files and directories at the same time in QFileDialog
        if False:
            l = od.findChild(QListView,"listView");
            if l is not None:
              l.setSelectionMode(QAbstractItemView.MultiSelection);

            t = od.findChild(QTreeView);
            if t is not None:
               t.setSelectionMode(QAbstractItemView.MultiSelection);
  
        nMode = od.exec()
        if nMode == QDialog.Accepted:
            _fnames = od.selectedFiles() # QStringList 
            print( "{}".format(_fnames) )
             
            try:
                if 1 == len(_fnames) and os.path.isdir(_fnames[0]):
                    self.loadData( _fnames[0] )
                elif 0 != len(_fnames):
                    self.statusBar().showMessage( "Formats other than MaLPi are not yet supported" )
            except Exception as ex:
                msg = "Error loading data: {}".format( str(ex) )
                self.statusBar().showMessage( msg )
                print( msg )

    def loadData(self, path):
        if not os.path.isdir(path):
            return
        self.data = DriveFormat.handlerForFile( path )
        if self.data is not None:
            self.path = path
            self.updateWindowTitle()
            self.initGrid()
            self.statusBar().showMessage( "{} images loaded".format( self.data.count() ) )
            self.slider.setMinimum(0)
            self.slider.setMaximum( self.data.count()-self.gridWidth )
            self.slider.setSliderPosition(0)
            self.updateImages()
        else:
            QMessageBox.warning(self, 'Unknown Filetype', "The file you selected could not be opened by any available file formats.", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)

    def saveData(self):
        if self.data is not None:
            self.data.save()
            self.statusBar().showMessage( "{} actions saved to {}".format( self.data.count(), self.path ) )
            self.updateWindowTitle()

    def closeEvent(self, event):
        if self.data is not None and not self.data.isClean():
            reply = QMessageBox.warning(self, 'Unsaved changes', "This document has unsaved changes.\nAre you sure you want to quit?", buttons=QMessageBox.Yes | QMessageBox.No, defaultButton=QMessageBox.No)
        else:
            reply = QMessageBox.Yes

        if reply == QMessageBox.Yes:
            self.data = None # For some reason we get here twice when quitting (as opposed to hitting escape), so clear this out
            event.accept()
        else:
            event.ignore()        

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_Left:
            if self.index > 0:
                self.index -= 1
                self.slider.setSliderPosition(self.index)
                self.updateImages()
            #else:
            #    print( "Not doing key left" )
        elif e.key() == Qt.Key_Right:
            if self.index < (self.data.count() - 1):
                self.index += 1
                self.slider.setSliderPosition(self.index)
                self.updateImages()
            #else:
            #    print( "Not doing key right" )
        elif e.key() == Qt.Key_Delete:
            if self.data is not None:
                self.data.deleteIndex(self.index)
                self.slider.setMaximum( self.data.count()-self.gridWidth )
                self.updateImages()
                self.updateStats()
        elif self.data is not None:
            newAction = self.data.actionForKey(e.text(),oldAction=self.data.actionForIndex(self.index))
            if newAction is None:
                e.ignore()
            else:
                self.changeCurrentAction( newAction )
        else:
            e.ignore()

    def changeCurrentAction(self, action, label_index=2):
        # Defaults to changing the action in the middle of the screen
        ot = self.data.outputTypes()
        if ot[0]["type"] == "categorical":
            self.actionLabels[label_index].setCurrentText( action )
        elif ot[0]["type"] == "continuous":
            self.actionLabels[label_index].setText( str(action) )
        if self.index >= self.data.count():
            print( "{} actions. index {}, label_index {}".format( self.data.count(), self.index, label_index ) )
        self.data.setActionForIndex( action, self.index )
        self.updateWindowTitle()
        self.updateStats()

    def mouseMoveEvent(self, e):
        #x = e.x()
        #y = e.y()

        #text = "x: {0},  y: {1}".format(x, y)
        #self.label.setText(text)
        pass

    def mousePressEvent(self, event):
        #self.c.closeApp.emit()
        pass

    def contextMenuEvent(self, event):
           cmenu = QMenu(self)
           
           newAct = cmenu.addAction("New")
           opnAct = cmenu.addAction("Open")
           quitAct = cmenu.addAction("Quit")
           # show and run the menu with exec_
           action = cmenu.exec_(self.mapToGlobal(event.pos()))
           
           if action == quitAct:
               qApp.quit()
        
    def centerWindowOnScreen(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def updateImages(self):
        for i in range( len(self.imageLabels) ):
            il = self.index - 2 + i
            if il >= 0 and il < self.data.count():
                image = self.data.imageForIndex( il )
                image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
                self.imageLabels[i].setPixmap( QPixmap(image) )
                self.indexes[i].setText( str( il ) )
                ot = self.data.outputTypes()
                if ot[0]["type"] == "categorical":
                    self.actionLabels[i].setCurrentText( self.data.actionForIndex( il ) )
                elif ot[0]["type"] == "continuous":
                    self.actionLabels[i].setText( str(self.data.actionForIndex( il )) )
                self.actionLabels[i].setEnabled(True)
            else:
                self.imageLabels[i].clear()
                self.indexes[i].setText( "" )
                self.actionLabels[i].setCurrentText( "" )
                self.actionLabels[i].setEnabled(False)

    def updateStats(self):
        if self.data is not None:
            stats = self.data.actionStats()
            self.statsTable.setRowCount(len(stats))
            self.statsTable.setColumnCount(2)
            row = 0
            for key in sorted(stats):
                value = stats[key]
                self.statsTable.setItem(row,0,QTableWidgetItem(key))
                self.statsTable.setItem(row,1,QTableWidgetItem(str(value)))
                row += 1


def runTests(args):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Adjust action values for a drive.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', nargs='*', metavar="File", help='Recorded drive data file to open')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    app = QApplication(sys.argv)
    ex = Example()
    if len(args.file) > 0:
        ex.loadData(args.file[0])
    sys.exit(app.exec_())
