""" A class for writing out everything associated with a training run.
E.g. Command line arguments, hyperparameters, input and output sizes, model description, results
"""
import os
import argparse
import datetime
import pickle

class Meta(object):
    def __init__(self, exp_name, args, exp_dir='experiments', num_samples=None, input_dim=None, num_actions=None, hparams={}):
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
            f.write( "Command line arguments:\n" )
            for key,value in vars(self.args).iteritems():
                f.write( "   {}: {}\n".format( key, value ) )
            f.write( "Hyperparameters:\n" )
            for key,value in self.hparams.iteritems():
                f.write( "   {}: {}\n".format( key, value ) )

    def writeAfter(self, model=None, histories=None, results={}):
        """ Write closing data to the experiment file.
            model: Needs to be a Keras model (with a summary method that accepts a print_fn argument)
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
            for key,value in results.iteritems():
                f.write( "   {}: {}\n".format( key, value ) )
        if histories is not None:
            his_fname = os.path.join(self.dir_name, "histories.pickle")
            with open(his_fname, 'wb') as f:
                pickle.dump( histories, f, pickle.HIGHEST_PROTOCOL)

def _hparamsTest():
    out = {}
    out["l2_reg"] = 0.000345
    out["learning_rate"] = 0.001
    out["batch_size"] = 64
    return out

def _runTests(args):
    try:
        exp1 = Meta( "TestExperimentName", args, exp_dir='test_experiments')
    except IOError as ex:
        print( "Caught correct exception when meta file already exists: PASS" )
    except Exception as exg:
        print( "Caught invalid exception ({}) when meta file already exists: FAIL".format(exg) )
    else:
        print( "No exception raised when meta file already exists: FAIL" )
    
    if args.file is None:
        n = datetime.datetime.now()
        testname = n.strftime('expMetaTest_%Y%m%d_%H%M%S')
    else:
        testname = args.file
    print( "testname: {}".format( testname ) )
    exp2 = Meta( testname, args, exp_dir="test_experiments", num_samples=10000, input_dim=(120,120,3), hparams=_hparamsTest() )

    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add( Dense(255, input_shape=(100,)) )
    results = {}
    results['val_acc'] = [0.3, 0.4, 0.5, 0.6]
    exp2.writeAfter( model=model, results=results )

def _getOptions():

    parser = argparse.ArgumentParser(description='Experiment Meta class and tests.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', help='Output file')
    parser.add_argument('--test_only', action="store_true", default=True, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _getOptions()

    if args.test_only:
        _runTests(args)
        exit()
