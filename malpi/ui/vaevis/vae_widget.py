import os
from pathlib import Path

from fastai.data.all import *
from fastai.vision.all import *
from fastai.data.transforms import ColReader, Normalize, RandomSplitter
from fastai.learner import load_learner

import torch
from torchvision import transforms as T
from PIL import Image

#from malpi.dk.train import preprocessFileList, get_data, get_learner, get_autoencoder, train_autoencoder
from malpi.dk.vae import VanillaVAE

from PyQt5.QtWidgets import QTextEdit, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QWidget, QAction, QMenu, QPushButton, QSlider
from PyQt5.QtWidgets import QDialog, QFileDialog, QDockWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

def load_vae_learner( file_path ):
    learn = load_learner( file_path )
    #learn.model.latent_dim
    return learn

class VAEDockWidget(QDockWidget):

    # Signals
    latentValueChanged = pyqtSignal(QImage)

    def __init__(self, title, parent):
        super().__init__(title, parent)
        
        self.vae = None
        self.mu = []
        self.var = []

        self.slider_scale = 100

        # From https://sparrow.dev/torchvision-transforms/
        self.preprocess = T.Compose([
            T.Resize(128), # TODO Get this from the loaded model
            T.ToTensor(),
            T.Normalize( mean=0.5, std=0.2 )
        ])

        self.reverse_preprocess = T.Compose([
            T.ToPILImage(),
            np.array,
        ])

        widg = QWidget(self)
        layout = QVBoxLayout(widg)

        self.loadModel = QPushButton('Load Model', self)
        self.loadModel.clicked.connect(self.handleLoadModel)

        pathWidg = QWidget(self)
        pathLayout = QHBoxLayout(pathWidg)
        self.pathLabel = QLabel("", pathWidg)
        pathLayout.addWidget(QLabel("Path: ", pathWidg))
        pathLayout.addWidget(self.pathLabel)
        
        layout.addWidget( pathWidg )

        self.filesTable = QTableWidget(self)
        self.filesTable.setEnabled(True)
        self.filesTable.horizontalHeader().setStretchLastSection(True)
        self.filesTable.horizontalHeader().hide()
        self.filesTable.verticalHeader().setDefaultSectionSize( 18 )
        self.filesTable.verticalHeader().hide()
        self.filesTable.setShowGrid(False)
        self.filesTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        #self.filesTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.filesTable.setSelectionMode(QAbstractItemView.NoSelection)
        #self.filesTable.cellDoubleClicked[int,int].connect(self.doubleClick)
        self.filesTable.setFocusPolicy(Qt.NoFocus)

        #self.filesTable.itemSelectionChanged.connect(self.selectionChanged)

        layout.addWidget(self.filesTable)

        self.setWidget(widg)
        #self.viewMenu.addAction( self.metaDock.toggleViewAction() )

    def preferredArea(self):
        return Qt.LeftDockWidgetArea


    def doubleClick( self, row, col ):
        if self.files is not None and row < len(self.files):
            newFile = self.files[row]
            if not newFile.startswith("#"):
                newFile = os.path.abspath(newFile).strip()
                self.newFileSelected.emit(newFile)

    def _selectedRow(self):
        model = self.filesTable.selectionModel()
        rows = model.selectedRows()
        if len(rows) > 0:
            return rows[0].row()
        return None

    def handleLoadModel(self):
        od = QFileDialog(self, 'Load Fastai VAE Model', "", "Pickled Models (*.pkl)")
        od.setAcceptMode(QFileDialog.AcceptOpen)
        #od.setFileMode(QFileDialog.Directory);
        od.setOption(QFileDialog.DontUseNativeDialog, True);

        nMode = od.exec()
        if nMode == QDialog.Accepted:
            _fnames = od.selectedFiles() # QStringList 
             
            try:
                if 1 == len(_fnames):
                    self.vae = load_vae_learner( _fnames[0] )
                    self.pathLabel.setText( _fnames[0] )
                    self.generateLatentControls()
                elif 0 != len(_fnames):
                    pass
            except Exception as ex:
                msg = "Error loading VAE Model: {}".format( str(ex) )
                #self.statusBar().showMessage( msg )
                print( msg )

    def generateLatentControls(self):
        latent = self.vae.latent_dim
        self.mu = [0] * latent
        self.var = [1] * latent
        self.sliders = []
        self.filesTable.setRowCount(latent)
        self.filesTable.setColumnCount(2)
        for row in range(latent):
            self.filesTable.setItem(row,0,QTableWidgetItem(str(row)))
            sld = QSlider(Qt.Horizontal, self.filesTable)
            sld.setMinimum( -self.var[row] * self.slider_scale )
            sld.setMaximum( self.var[row] * self.slider_scale )
            sld.setSliderPosition( self.mu[row] )
            sld.setFocusPolicy(Qt.NoFocus)
            sld.valueChanged[int].connect(self.zSliderChangedValue)
            sld.setProperty("zdim", row)
            self.filesTable.setCellWidget(row,1,sld)
            self.sliders.append( sld )
        #self.filesTable.selectRow(0)

    def zSliderChangedValue(self, value):
        float_value = value / self.slider_scale
        zdim = self.sender().property("zdim")
        self.mu[zdim] = float_value
        #print( f"Z value: {zdim} {float_value}" )
        T = torch.tensor(self.mu)
        samples = self.vae.model.decode(T)

        image = samples[0]

        try:
            out_tensor = torch.squeeze(image)
            p_image = self.reverse_preprocess(out_tensor)
            o_image = QImage(p_image, p_image.shape[1], p_image.shape[0], QImage.Format_RGB888)

            self.latentValueChanged.emit(o_image)
        except Exception as ex:
            print( f"QImage error: {ex}" )

    def setImage(self, pil_image):
        try:
            img_tensor = self.preprocess(pil_image)
            img_tensor = torch.unsqueeze( img_tensor, 0 )
            out_tensor, in_tensor, mu, log_var = self.vae.model.forward(img_tensor)


            """ Copy model's mu values into self.mu and reset all QSliders. """
            self.mu = torch.squeeze(mu).detach().numpy()
            self.log_var = torch.squeeze(log_var).detach().numpy()
            print( f"mu: {type(mu)} {mu.shape}" )
            for idx, z in enumerate(self.mu):
                self.sliders[idx].setSliderPosition(z)
                self.var[idx] = self.log_var[idx]
                var = self.var[idx]
                if var < 0.1:
                    var = 0.1
                self.sliders[idx].setMinimum( -var * self.slider_scale )
                self.sliders[idx].setMaximum( var * self.slider_scale )
        except Exception as ex:
            print( f"VAEDockWidget.setImage" )
            raise

        try:
            out_tensor = torch.squeeze(out_tensor)
            p_image = self.reverse_preprocess(out_tensor)
            o_image = QImage(p_image, p_image.shape[1], p_image.shape[0], QImage.Format_RGB888)
            self.latentValueChanged.emit(o_image)
        except Exception as ex:
            print( f"QImage error: {ex}" )
