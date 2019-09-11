""" A class for writing out everything associated with a training run.
E.g. Command line arguments, hyperparameters, input and output sizes, model description, results
"""
import os
import sys
import argparse
import datetime
import pickle
import subprocess

class Experiment(object):

    __version__ = "1.0.0"

    def __init__(self, exp_name, args, exp_dir='experiments', num_samples=None, input_dim=None, num_actions=None, hparams={}, modules=[]):
        """ modules: iterable with python modules that have __name__ and __version__ attributes.
                Python and Experiment class versions will always be output.
        """
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        self.dir_name = os.path.join( exp_dir, exp_name )
        if os.path.exists(self.dir_name):
            raise IOError(-1,"Experiment directory already exists, won't overwrite ",self.dir_name) 

        os.makedirs(self.dir_name)

        self.args = args
        self.filename = os.path.join( self.dir_name, exp_name+".txt" )
        self.name = exp_name
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hparams = hparams
        self.modules = modules
        self.commit = Experiment.gitCommit()
        self.start = datetime.datetime.now()
        self.writeBefore()

    def _writeOne(self, fileobj, key, value, indent=''):
        if value is not None:
            fileobj.write( "{}{}: {}\n".format( indent, key, value ) )

    def writeBefore(self):
# Write out everything we know before running the experiment, in case of crash, etc.
# Will overwrite any existing file
        with open(self.filename,'w') as f:
            f.write( "Name: {}\n".format( self.name ) )
            f.write( "Start: {}\n".format( self.start ) )
            self._writeOne( f, "Num samples", self.num_samples )
            self._writeOne( f, "Input dims", self.input_dim )
            self._writeOne( f, "Num actions", self.num_actions )
            if self.commit is None:
                print( "Commit: None" )
            self._writeOne( f, "Git Commit", self.commit )
            f.write( "Command line arguments:\n" )
            for key,value in vars(self.args).items():
                self._writeOne( f, key, value, indent="   ")
            f.write( "Hyperparameters:\n" )
            for key,value in self.hparams.items():
                self._writeOne( f, key, value, indent="   ")
            self._writeVersions(f)

    def writeAfter(self, model=None, histories=None, results={}, saveModel=False):
        """ Write closing data to the experiment file.
            model: Needs to be a Keras model (with a summary method that accepts a print_fn argument)
                It also needs to support to_json() and save_weights() methods if saveModel is True.
            results: A dictionary of any relevant results
        """
# Write out everything new we know after running the experiment
# Will append to the existing file
        with open(self.filename,'a') as f:
            finish = datetime.datetime.now()
            f.write( "Finish: {}\n".format( finish ) )
            f.write( "Elapsed: {}\n".format( finish-self.start ) )
            if model is not None:
                summ_list = []
                model.summary(print_fn=lambda x: summ_list.append(x))
                f.write( "Model:\n" )
                for summ in summ_list:
                    f.write( '   {}\n'.format(summ) )
            f.write( "Results:\n" )
            for key,value in results.items():
                f.write( "   {}: {}\n".format( key, value ) )
        if histories is not None:
            his_fname = os.path.join(self.dir_name, "histories.pickle")
            with open(his_fname, 'wb') as f:
                pickle.dump( histories, f, pickle.HIGHEST_PROTOCOL)
        if model is not None and saveModel:
            fname = os.path.join( self.dir_name, self.name+"_model.json" )
            with open(fname,'w') as f:
                f.write(model.to_json())
            fname = os.path.join( self.dir_name, self.name+"_weights.h5" )
            model.save_weights(fname)

    def pythonVersionString(self):
        """Current system python version as string major.minor.micro [(alpha|beta|etc)]"""
        vstring = "{0}.{1}.{2}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
        if sys.version_info.releaselevel != "final":
            vstring += " ({})".format( sys.version_info.releaselevel )
        if sys.version_info.serial != 0:
            vstring += " (serial: {})".format( sys.version_info.serial )
        return vstring

    def _writeVersions(self, fileobj):
        fileobj.write( "Module Versions:\n" )
        self._writeOne( fileobj, "Python", self.pythonVersionString(), indent="   " )
        self._writeOne( fileobj, "Experiment", Experiment.__version__, indent="   " )

        for mod in self.modules:
            self._writeOne( fileobj, mod.__name__, mod.__version__, indent="   " )

        self.modules = []

    @staticmethod
    def gitCommit():
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if 0 == out.returncode:
                return out.stdout.decode('UTF-8').strip()
        except Exception as ex:
            return None
        return None

