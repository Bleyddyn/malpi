#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" A tool for modifying data collected by MaLPi.
An algorithm called DAgger [need ref] involves collecting data while an agent/robot is running its policy,
then having an 'expert' go over the data and correct any mistakes the agent made,
then retraining on the cleaned up data.

TODO:
1) Add in-app help (keyboard shortcuts, etc)
2) Add currently open filename to window (text field or window title)
3) Make it obvious which action will be changed via keyboard shortcut
4) Pre-generate QImages and store them
5) Load/Save DonkeyCar tub files
6) Add an indicator if the file has been changed and warn the user if they try to exit without saving
"""

import sys
import os
import argparse

from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, QMenu, QMessageBox, QApplication, QDesktopWidget, qApp, QPushButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QSlider, QListView, QTreeView, QAbstractItemView, QComboBox
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject

import dagger_data
import DriveFormat

class Communicate(QObject):
    closeApp = pyqtSignal() 

class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.data = None
        self.index = 0
        self.gridWidth = 5

        self.initUI()

    def initUI(self):               

        wid = QWidget(self)
        self.setCentralWidget(wid)

        self.c = Communicate()
        self.c.closeApp.connect(self.close)       
        
        #self.initButtons()
        #self.setMouseTracking(True)

        self.setWindowTitle('DAgger Tool')    
        self.statusBar().showMessage('Ready')
        self.initMenus()

        self.setGeometry(200, 200, 800, 600)
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
        
        viewStatAct = QAction('View statusbar', self, checkable=True)
        viewStatAct.setStatusTip('View statusbar')
        viewStatAct.setChecked(True)
        viewStatAct.triggered.connect(self.toggleMenu)
        
        viewMenu.addAction(viewStatAct)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAct)
        self.toolbar.addAction(openFile)

    def initButtons(self):
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.centralWidget().setLayout(vbox)

    def makeActionComboBox(self):
        ae = QComboBox(self)
        if self.data is not None:
            for aclab in self.data.actionNames():
                ae.addItem(aclab)
        ae.setInsertPolicy(QComboBox.NoInsert)
        ae.activated[str].connect(self.actionEdited)
        return ae

    def initGrid(self):
        self.imageLabels = []
        self.actionLabels = []
        self.indexes = []
        for i in range(self.gridWidth):
            self.imageLabels.append( QLabel(self) )
            self.actionLabels.append( self.makeActionComboBox() )
            idx = QLabel(self)
            idx.setAlignment( Qt.AlignCenter )
            self.indexes.append( idx )

        sld = QSlider(Qt.Horizontal, self)
        self.slider = sld
        sld.setFocusPolicy(Qt.NoFocus)
        sld.valueChanged[int].connect(self.changeValue)


        grid = QGridLayout()
        grid.setSpacing(10)

        row = 1
        grid.addWidget(QLabel('Images'), row, 0)
        for i in range(len(self.imageLabels)):
            grid.addWidget(self.imageLabels[i], row, i+1)

        row += 1
        grid.addWidget(QLabel('Actions'), row, 0)
        for i in range(len(self.actionLabels)):
            grid.addWidget(self.actionLabels[i], row, i+1)

        row += 1
        for i in range(len(self.indexes)):
            grid.addWidget(self.indexes[i], row, i+1)

        row += 1
        grid.addWidget(sld, row, 0, 1, self.gridWidth+1)

        self.centralWidget().setLayout(grid)

    def changeValue(self, value):
        self.index = value
        self.updateImages()

    def actionEdited(self, newValue):
        idx = self.actionLabels.index(self.sender())
        self.data.setActionForIndex( newValue, self.index + idx )

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
        self.data = DriveFormat.Drive(path)
        self.path = path
        self.initGrid()
        self.statusBar().showMessage( "{} images loaded".format( self.data.count() ) )
        self.slider.setMinimum(0)
        self.slider.setMaximum( self.data.count()-self.gridWidth )
        self.slider.setSliderPosition(0)
        self.updateImages()

    def saveData(self):
        if self.data is not None:
            self.data.save()
            self.statusBar().showMessage( "{} actions saved to {}".format( self.data.count(), self.path ) )

    def closeEvent(self, event):
        if self.data is not None and not self.data.isClean():
            reply = QMessageBox.warning(self, 'Unsaved changes', "This document has unsaved changes.\nAre you sure you want to quit?", buttons=QMessageBox.Yes | QMessageBox.No, defaultButton=QMessageBox.No)
        else:
            reply = QMessageBox.Yes

        if reply == QMessageBox.Yes:
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
            if self.index < (self.data.count() - self.gridWidth):
                self.index += 1
                self.slider.setSliderPosition(self.index)
                self.updateImages()
            #else:
            #    print( "Not doing key right" )

        if self.data is not None:
            newAction = self.data.actionForKey(e.text())
            if newAction is None:
                e.ignore()
            else:
                self.changeCurrentAction( newAction )
        else:
            e.ignore()

    def changeCurrentAction(self, action, label_index=2):
        # Defaults to changing the action in the middle of the screen
        self.actionLabels[label_index].setCurrentText( action )
        if (self.index+label_index) >= self.data.count():
            print( "{} actions. index {}, label_index {}".format( self.data.count(), self.index, label_index ) )
        self.data.setActionForIndex( action, self.index+label_index )

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
            if self.index + i < self.data.count():
                image = self.data.imageForIndex( self.index + i )
                image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
                self.imageLabels[i].setPixmap( QPixmap(image) )
                self.actionLabels[i].setCurrentText( self.data.actionForIndex( self.index + i ) )
                self.indexes[i].setText( str( self.index + i ) )
            else:
                print( "Not doing updateImages for index {} + {}".format( self.index, i ) )

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

#def save( self ):
#def isClean( self ):
#def imageForIndex( self, index ):
#def actionForIndex( self, index ):
#def setActionForIndex( self, action, index ):
#def actionNames():
