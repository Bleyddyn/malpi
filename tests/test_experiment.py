""" Some simple tests for MaLPi's Experiment tracking class.
    TODO: rewrite so it can be used by pytest.
"""

import argparse
import datetime

from malpi import Experiment

def _hparamsTest():
    out = {}
    out["l2_reg"] = 0.000345
    out["learning_rate"] = 0.001
    out["batch_size"] = 64
    return out

def _runTests(args):
    try:
        exp1 = Experiment( "TestExperimentName", args, exp_dir='test_experiments')
    except IOError as ex:
        print( "Caught correct exception when meta file already exists: PASS" )
    except Exception as exg:
        print( "Caught invalid exception ({}) when meta file already exists: FAIL".format(exg) )
    else:
        print( "No exception raised when meta file already exists: FAIL" )
    
    import keras
    import tensorflow as tf
    import numpy as np

    if args.file is None:
        n = datetime.datetime.now()
        testname = n.strftime('expMetaTest_%Y%m%d_%H%M%S')
    else:
        testname = args.file
    print( "testname: {}".format( testname ) )
    exp2 = Experiment( testname, args, exp_dir="test_experiments", num_samples=10000, input_dim=(120,120,3), hparams=_hparamsTest(), modules=[np, tf, keras] )

    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add( Dense(255, input_shape=(100,)) )
    results = {}
    results['val_acc'] = [0.3, 0.4, 0.5, 0.6]
    exp2.writeAfter( model=model, results=results )

def _getOptions():

    parser = argparse.ArgumentParser(description='Experiment class and tests.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', help='Output file')
    parser.add_argument('--test_only', action="store_true", default=True, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _getOptions()

    if args.test_only:
        _runTests(args)
        exit()
