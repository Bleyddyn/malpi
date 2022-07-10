#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" A tool for visualizing a PyTorch VAE.
"""

import sys
import os
import argparse

from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, QMenu, QMessageBox, QApplication, QDesktopWidget, qApp, QPushButton, QSizePolicy
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QSlider, QListView, QTreeView, QAbstractItemView, QComboBox, QFrame
from PyQt5.QtWidgets import QDialog, QFileDialog, QDockWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtCore import pyqtSignal, QObject

from torchvision import transforms
from PIL import Image
import numpy as np

from malpi.ui.DriveFormat import DriveFormat
# Import all formats so they can register themselves
import malpi.ui.MalpiFormat
import malpi.ui.TubFormat
import malpi.ui.Tubv2Format
from malpi.ui.vaevis.vae_widget import VAEDockWidget

class Communicate(QObject):
    closeApp = pyqtSignal() 

class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.data = None
        self.aux = []
        self.index = 0
        self.gridWidth = 5
        self.viewMenu = None
        self.metaDock = None
        self.statsDock = None
        self.modelDock = None
        self.path = ""

        self.initUI()

    def initUI(self):               

        wid = QWidget(self)
        self.setCentralWidget(wid)
        vlay = QHBoxLayout()
        self.centralWidget().setLayout(vlay)

        self.c = Communicate()
        self.c.closeApp.connect(self.close)       
        
        #self.setMouseTracking(True)

        self.updateWindowTitle()
        self.statusBar().showMessage('Ready')
        self.initMenus()

        if self.modelDock is None:
            self.modelDock = VAEDockWidget("Model", self)
            self.addDockWidget(self.modelDock.preferredArea(), self.modelDock)
            self.viewMenu.addAction( self.modelDock.toggleViewAction() )
            self.modelDock.setVisible(True)
            self.modelDock.latentValueChanged.connect(self.generateImage)

        source_box = QLabel(wid)
        sb_layout = QVBoxLayout()
        source_box.setLayout(sb_layout)
        vlay.addWidget( source_box )
        source_label = QLabel("Original Image", source_box)
        sb_layout.addWidget( source_label )
        self.sourceImage = QLabel(source_box)
        self.sourceImage.setSizePolicy( QSizePolicy.Expanding, QSizePolicy.Expanding )
        sb_layout.addWidget(self.sourceImage)

        dest_box = QLabel(wid)
        dest_layout = QVBoxLayout()
        dest_box.setLayout(dest_layout)
        vlay.addWidget( dest_box )
        dest_label = QLabel("Reconstruction/Sample", dest_box)
        dest_layout.addWidget( dest_label )
        self.outputImage = QLabel(dest_box)
        self.outputImage.setSizePolicy( QSizePolicy.Expanding, QSizePolicy.Expanding )
        dest_layout.addWidget(self.outputImage)

        # Use all available space
        #self.setGeometry(QDesktopWidget().availableGeometry())
        self.resize(QApplication.primaryScreen().availableSize() * 4 / 5);

        self.show()
        
    def initMenus(self):
        #exitAct = QAction(QIcon('exit.png'), 'Exit', self)        
        #exitAct = QAction(QIcon('exit24.png'), 'Exit', self)
        exitAct = QAction('Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.quit)

        openFile = QAction('Load Image', self)
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

        self.dataMenu = menubar.addMenu('Data')

        self.auxDataAct = QAction('Add Auxiliary Data', self)
        self.auxDataAct.setStatusTip('Add Auxiliary Data to the current data file')
        self.auxDataAct.triggered.connect(self.addAuxData)
        
        self.dataMenu.addAction(self.auxDataAct)

        nextDone = QAction('Find Next Done', self)
        nextDone.setStatusTip('Find the next sample with a Done flag set to True')
        nextDone.setShortcut('Ctrl+D')
        nextDone.triggered.connect(self.findNextDone)
        self.dataMenu.addAction(nextDone)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAct)
        self.toolbar.addAction(openFile)

    def makeActionComboBox(self, atype, labels=None):
        ae = None

        if atype == "categorical":
            ae = QComboBox(self)
            for aclab in labels:
                ae.addItem(aclab)
            ae.setInsertPolicy(QComboBox.NoInsert)
            ae.activated[str].connect(self.actionEdited)
        elif atype == "continuous":
            ae = QLineEdit(self)
            ae.setEnabled(False)
            ae.setReadOnly(True)
            ae.setFocusPolicy(Qt.NoFocus)
            ae.installEventFilter(self)

        if ae is None:
            ae = QLineEdit(self)
        return ae

    def initGrid(self):
        if self.data is None:
            return

        # This needs to be generalized to handle multiple input and output types
        self.imageLabels = []
        self.actionRows = []
        self.actionLabels = []

        for i in range(self.gridWidth):
            self.imageLabels.append( QLabel(self) )

        ot = self.data.outputTypes()
        for output in ot:
            oneRow = []
            for i in range(self.gridWidth):
                labels = output["categories"] if "categories" in output else None
                cb = self.makeActionComboBox( atype=output["type"], labels=labels)
                oneRow.append( cb )
                self.actionLabels.append( cb )
            self.actionRows.append(oneRow)

        grid = QGridLayout()
        self.grid = grid
        grid.setSpacing(10)

        otypes = self.data.outputTypes()
        itypes = self.data.inputTypes()

        row = 1
        grid.addWidget(QLabel("Index"), row, 0)
        self.indexes = []
        for i in range(self.gridWidth):
            idx = QLabel(self)
            idx.setAlignment( Qt.AlignCenter )
            self.indexes.append( idx )
        for i in range(len(self.indexes)):
            grid.addWidget(self.indexes[i], row, i+1)

        row += 1
        grid.addWidget(QLabel(itypes[0]["name"]), row, 0)
        for i in range(len(self.imageLabels)):
            grid.addWidget(self.imageLabels[i], row, i+1)

        for lr in range(len(self.actionRows)):
            arow = self.actionRows[lr]
            row += 1
            grid.addWidget(QLabel(otypes[lr]["name"]), row, 0)
            for i in range(len(arow)):
                grid.addWidget(arow[i], row, i+1)

        vlay = QVBoxLayout()
        vlay.addLayout(grid)
        row += 1
        sld = QSlider(Qt.Horizontal, self)
        self.slider = sld
        sld.setFocusPolicy(Qt.NoFocus)
        sld.valueChanged[int].connect(self.changeValue)
        #grid.addWidget(sld, row, 0, 1, self.gridWidth+1)
        vlay.addWidget(sld)

        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(vlay)

        if self.metaDock is None:
            self.metaDock = QDockWidget("Drive Info", self)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.metaDock)
            self.metaText = QTextEdit(self)
            self.metaText.setEnabled(False)
            self.metaText.setReadOnly(True)
            #self.metaText.setMinimumWidth( 200.0 )
            self.metaDock.setWidget(self.metaText)
            self.viewMenu.addAction( self.metaDock.toggleViewAction() )
        self.metaText.setText( self.data.metaString() )

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
        self.setWindowTitle('VAE Visualizer' + path + edit_msg)

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

    def quit(self):
        if self.data is not None and not self.data.isClean():
            reply = self.unsavedDataAskUser("This document has unsaved changes.\nAre you sure you want to quit?", "Quit", "Don't Quit")
            if not reply:
                return
        qApp.quit()

    def openDialog(self):
        od = QFileDialog(self, 'Load an image and set z-space values', "", "Images (*.jpg *.jpeg *.tif *.tiff *.png)")
        od.setAcceptMode(QFileDialog.AcceptOpen)
        od.setFileMode(QFileDialog.ExistingFile);
        od.setOption(QFileDialog.DontUseNativeDialog, True);

        nMode = od.exec()
        if nMode == QDialog.Accepted:
            _fnames = od.selectedFiles() # QStringList 
             
            try:
                if 1 == len(_fnames):
                    self.loadImage( _fnames[0] )
            except Exception as ex:
                msg = "Error loading data: {}".format( str(ex) )
                self.statusBar().showMessage( msg )
                print( msg )

    def unsavedDataAskUser(self, text, yesButtonText, noButtonText):
        msgBox = QMessageBox(QMessageBox.Warning, "Unsaved changes", text)
        #msgBox.setTitle("Unsaved changes")
        #msgBox.setText(text)
        pButtonYes = msgBox.addButton(yesButtonText, QMessageBox.YesRole)
        msgBox.addButton(noButtonText, QMessageBox.NoRole)

        msgBox.exec()

        if msgBox.clickedButton() == pButtonYes:
            return True

        return False

    def loadImage(self, path):
        # display it
        image = QImage(path)
        self.sourceImage.setPixmap( QPixmap(image) )

        # pass it to the model to encode and set slider values
        self.modelDock.setImage(Image.open(path))

    def saveData(self):
        if self.data is not None:
            self.data.save()
            self.statusBar().showMessage( "{} actions saved to {}".format( self.data.count(), self.path ) )
            self.updateWindowTitle()

    def closeEvent(self, event):
        if self.data is not None and not self.data.isClean():
            reply = self.unsavedDataAskUser("This document has unsaved changes.\nAre you sure you want to quit?", "Quit", "Don't Quit")
        else:
            reply = False

        if reply:
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
        elif e.key() == Qt.Key_Right:
            if self.data is not None:
                if self.index < (self.data.count() - 1):
                    self.index += 1
                    self.slider.setSliderPosition(self.index)
                    self.updateImages()
        elif e.key() == Qt.Key_Down:
            pass
        elif e.key() == Qt.Key_Up:
            pass
        elif e.key() == Qt.Key_Return:
            pass
        elif e.key() == Qt.Key_Delete:
            if self.data is not None:
                self.data.deleteIndex(self.index)
                self.slider.setMaximum( self.data.count() )
                self.slider.setSliderPosition(self.index)
                self.updateImages()
                self.updateStats()
                self.updateWindowTitle()
        elif self.data is not None:
            if self.index < (self.data.count() - 1):
                newAction = self.data.actionForKey(e.text(),oldAction=self.data.actionForIndex(self.index))
                if newAction is None:
                    for auxUI in self.aux:
                        if auxUI.handleKeyPressEvent( e, self.index):
                            e.accept()
                            return
                    e.ignore()
                else:
                    self.changeCurrentAction( newAction )
        else:
            e.ignore()


    def eventFilter( self, obj, event ):
        if event.type() == QEvent.MouseButtonDblClick:
            if obj in self.actionLabels:
                obj.setEnabled(True)
                obj.setReadOnly(False)
        return False;

    def changeCurrentAction(self, action, label_index=2):
        # Defaults to changing the action in the middle of the screen
        self.data.setActionForIndex( action, self.index )
        ot = self.data.outputTypes()
        if ot[0]["type"] == "categorical":
            self.actionLabels[label_index].setCurrentText( action )
        elif ot[0]["type"] == "continuous":
            #self.actionLabels[label_index].setText( str(action) )
            self.updateImages()
        if self.index >= self.data.count():
            print( "{} actions. index {}, label_index {}".format( self.data.count(), self.index, label_index ) )
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
           
           #newAct = cmenu.addAction("New")
           opnAct = cmenu.addAction("Open")
           quitAct = cmenu.addAction("Quit")
           # show and run the menu with exec_
           action = cmenu.exec_(self.mapToGlobal(event.pos()))
           
           if action == quitAct:
               self.quit()
           elif action == opnAct:
               self.openDialog()

    def generateImage(self, image):
        self.outputImage.setPixmap( QPixmap(image) )
        pass

    def updateImages(self):
        for i in range( len(self.imageLabels) ):
            il = self.index - 2 + i
            ot = self.data.outputTypes()
            if il >= 0 and il < self.data.count():
                image = self.data.imageForIndex( il )
                image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
                self.imageLabels[i].setPixmap( QPixmap(image) )
                self.indexes[i].setText( str( il ) )

                actions = self.data.actionForIndex( il )
                for oi in range(len(ot)):
                    oneRow = self.actionRows[oi]
                    oneO = ot[oi]
                    if oneO["type"] == "categorical":
                        oneRow[i].setCurrentText( actions[oi] )
                    elif oneO["type"] == "continuous":
                        oneRow[i].setText( str(actions[oi]) )
                    oneRow[i].setEnabled(True)
            else:
                self.imageLabels[i].clear()
                self.indexes[i].setText( "" )
                if ot[0]["type"] == "categorical":
                    self.actionLabels[i].setCurrentText( "" )
                elif ot[0]["type"] == "continuous":
                    self.actionLabels[i].setText( "" )
                self.actionLabels[i].setEnabled(False)
        for auxUI in self.aux:
            auxUI.update(self.index)

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

    def handleAuxDataChanged(self, auxName):
        # May not need this
        self.updateImages()

    def addAuxToGrid(self, aux):
        try:
            if not self.data.supportsAuxData():
                return
            auxUI = AuxDataUI.create(aux, self.data, self.gridWidth)
            self.aux.append(auxUI)
            auxName, auxLabels = auxUI.getUI()

            row = self.grid.rowCount()
            self.grid.addWidget(auxName, row, 0)
            for i, auxLabel in enumerate(auxLabels):
                self.grid.addWidget(auxLabel, row, i+1)

            auxUI.auxDataChanged[str].connect(self.handleAuxDataChanged)
        except Exception as ex:
            print( "Error loading auxiliary data: {}".format( ex ) )

    def addAuxData(self):
        if self.data is not None:
            if self.data.supportsAuxData():
                dlg = AuxDataDialog(self)
                nMode = dlg.exec()
                if nMode == QDialog.Accepted:
                    print( "Meta: {}".format( dlg.getMeta() ) )
                    self.addAuxToGrid(dlg.getMeta())
                else:
                    print( "Cancelled, don't create auxiliary data" )

    def findNextDone(self):
        for i in range(self.index+1,self.data.count()):
            if self.data.isIndexDeleted(i):
                continue
            data = self.data.auxDataAtIndex( "done", i )
            if data == "True":
                self.index = i
                self.slider.setSliderPosition(self.index)
                self.updateImages()
                break

def runTests(args):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Adjust action values for a drive.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', nargs='*', metavar="File", help='Recorded drive data file to open')
    parser.add_argument('-f', '--file', dest="filelist", help='File with a list of files to open, one per line')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    return args

def main():

    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
