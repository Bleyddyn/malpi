#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial 

This program shows a confirmation 
message box when we click on the close
button of the application window. 

Author: Jan Bodnar
Website: zetcode.com 
Last edited: August 2017
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, QMenu, QMessageBox, QApplication, QDesktopWidget, qApp, QPushButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QSlider
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject

class Communicate(QObject):
    closeApp = pyqtSignal() 

class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        
    def initUI(self):               

        wid = QWidget(self)
        self.setCentralWidget(wid)

        self.c = Communicate()
        self.c.closeApp.connect(self.close)       
        
        #self.initButtons()
        self.initGrid()

        self.setWindowTitle('Message box')    
        self.statusBar().showMessage('Ready')
        self.initMenus()

        self.setGeometry(300, 300, 250, 150)
        self.center()

        self.show()
        
    def initMenus(self):
        #exitAct = QAction(QIcon('exit.png'), 'Exit', self)        
        #exitAct = QAction(QIcon('exit24.png'), 'Exit', self)
        exitAct = QAction('Exit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        openFile = QAction('Open', self)
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        impAct = QAction('Import File', self)
        impMenu = QMenu('Import', self)
        impMenu.addAction(impAct)

        fileMenu = menubar.addMenu('File')
        fileMenu.addMenu(impMenu)
        fileMenu.addAction(openFile)
        fileMenu.addAction(exitAct)

        viewMenu = menubar.addMenu('View')
        
        viewStatAct = QAction('View statusbar', self, checkable=True)
        viewStatAct.setStatusTip('View statusbar')
        viewStatAct.setChecked(True)
        viewStatAct.triggered.connect(self.toggleMenu)
        
        viewMenu.addAction(viewStatAct)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAct)

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

    def initGrid(self):
        title = QLabel('Image')
        author = QLabel('Author')
        review = QLabel('Review')

        self.imageLabel = QLabel(self)
        self.imageLabel.setPixmap(QPixmap('mute.png')) # needs to have a path to an existing image

        titleEdit = self.imageLabel

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.valueChanged[int].connect(self.changeValue)

        self.reviewEdit = QTextEdit()

        grid = QGridLayout()
        grid.setSpacing(10)

        row = 1
        self.setMouseTracking(True)
        self.label = QLabel('Mouse')
        grid.addWidget( QLabel('Mouse'), row, 0)
        grid.addWidget( self.label, row,1)

        row += 1
        grid.addWidget(title, row, 0)
        grid.addWidget(titleEdit, row, 1)

        row += 1
        grid.addWidget(sld, row, 0, 1, 2)

        row += 1
        grid.addWidget(review, row, 0)
        grid.addWidget(self.reviewEdit, row, 1, 5, 1)

        self.centralWidget().setLayout(grid)

    def changeValue(self, value):
        if value == 0:
            self.imageLabel.setText('mute')
        elif value > 0 and value <= 30:
            self.imageLabel.setText('low')
        elif value > 30 and value < 80:
            self.imageLabel.setText('normal')
        else:
            self.imageLabel.setText('high')

    def toggleMenu(self, state):
        if state:
            self.statusBar().show()
        else:
            self.statusBar().hide()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                self.reviewEdit.setText(data)        
        
    def closeEvent(self, event):
        #reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        reply = QMessageBox.Yes

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()        

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

        text = "x: {0},  y: {1}".format(x, y)
        self.label.setText(text)

    def mousePressEvent(self, event):
        self.c.closeApp.emit()

    def contextMenuEvent(self, event):
           cmenu = QMenu(self)
           
           newAct = cmenu.addAction("New")
           opnAct = cmenu.addAction("Open")
           quitAct = cmenu.addAction("Quit")
           # show and run the menu with exec_
           action = cmenu.exec_(self.mapToGlobal(event.pos()))
           
           if action == quitAct:
               qApp.quit()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
 
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
