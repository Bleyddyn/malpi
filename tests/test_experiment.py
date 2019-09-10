""" Some simple tests for MaLPi's Experiment tracking class.
    Usage: python3 -m pytest
"""

import argparse
import datetime
import pytest
import shutil

from malpi import Experiment

def _hparamsTest():
    out = {}
    out["l2_reg"] = 0.000345
    out["learning_rate"] = 0.001
    out["batch_size"] = 64
    return out

def _setup():
    expdir = "exp_test_dir1"
    try:
        shutil.rmtree(expdir)
    except FileNotFoundError as ex:
        pass
    return expdir

def test_exists():
    expdir = _setup()
    args = _getOptions("--file=TestFile.txt".split())

    exp1 = Experiment( "TestExperimentName1", args, exp_dir=expdir)
    with pytest.raises(IOError) as e_info:
        exp2 = Experiment( "TestExperimentName1", args, exp_dir=expdir)

    # Shouldn't raise an exception because of the different exp name
    exp2 = Experiment( "TestExperimentName2", args, exp_dir=expdir)

def test_experiment():
    expdir = _setup()
    args = _getOptions("--file=TestFile.txt".split())
    
# modules that should always be available and have name and version attributes.
    import setuptools as mod1
    import pip as mod2
    import numpy as mod3

    n = datetime.datetime.now()
    testname = n.strftime('expMetaTest_%Y%m%d_%H%M%S')
    exp2 = Experiment( testname, args, exp_dir=expdir, num_samples=10000, input_dim=(120,120,3), hparams=_hparamsTest(), modules=[mod1, mod2, mod3] )

    results = {}
    results['val_acc'] = [0.3, 0.4, 0.5, 0.6]
    exp2.writeAfter( model=None, results=results )

def _getOptions(argstring=None):

    parser = argparse.ArgumentParser(description='Experiment class and tests.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', help='Output file')
    parser.add_argument('--test_only', action="store_true", default=True, help='run tests, then exit')

    args = parser.parse_args(argstring)

    return args
