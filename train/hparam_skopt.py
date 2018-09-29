import pickle
import datetime
import numpy as np
from skopt.space import Real, Integer, Categorical, Dimension, Identity
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import dump, load

# Requires the latest version:
# pip3 install git+https://github.com/scikit-optimize/scikit-optimize/
from skopt.callbacks import CheckpointSaver

from train import fitFC, setCPUCores, hparamsToDict, hparamsToArray
from load_drives import DriveDataGenerator

import keras.backend as K

class Holder:
    def __init__(self, dirs, log_to=None):
        self.dirs = dirs
        self.vals = []
        self.run = 0
        self.log_to = log_to
        self.input_dim = (120,120,3)
        self.error_value = 10000

    def _makeGenerators(self, dirs, batch_size):
        val_split = 0.2
        last = int(len(dirs) * val_split)
        np.random.shuffle(dirs)
        test = dirs[:last]
        train = dirs[last:]
        dirs = train
        val = test
        image_size = (120,120)
        gen = DriveDataGenerator(dirs, image_size=image_size, batch_size=batch_size, shuffle=True, max_load=20000, auxName=None )
        val = DriveDataGenerator(val, image_size=image_size, batch_size=batch_size, shuffle=True, max_load=10000, auxName=None )
        #cat = gen.categorical
        return gen, val

#    @use_named_args(space)
    def __call__(self, args):
        try:
            print( args )
            l2_reg, dropouts, learning_rate, batch_size, optimizer = args
            arg_dict = { 'l2_reg': l2_reg,
                         'dropouts': dropouts,
                         'learning_rate': learning_rate,
                         'batch_size': batch_size,
                         'optimizer': optimizer }

            self.run += 1
            hparams = hparamsToDict( hparamsToArray( arg_dict ) )

            gen, val = self._makeGenerators(self.dirs, batch_size)
            num_actions = gen.num_actions
            num_samples = gen.count

            print( "Run {}".format( self.run ) )
            print( "   Args {}".format( args ) )
            print( "   Hparams {}".format( hparams ) )
            val, his, model = fitFC( self.input_dim, num_actions, gen, val, val_set=None, verbose=0, dkconv=True, early_stop=True,
                    categorical=gen.categorical, pre_model=None, **hparams )
            self.vals.append(val)
            print( "   Val acc {}".format( val ) )
            if self.log_to is not None:
                with open( self.log_to, 'a' ) as f:
                    f.write( "Run {}".format( self.run ) )
                    f.write( "   Hparams {}".format( hparams ) )
                    f.write( "   Val loss {}\n".format( val ) )
            return val
        except:
            return self.error_value

def previousRuns(filename):

    data = load(filename)

    x0 = data['x_iters']
    y0 = data['func_vals']

    return x0, y0

def runParamTests(args):
    best = {'timesteps': 7.0, 'learning_rate': 0.001306236693845287, 'batch_size': 8.0}
    best = {'timesteps': 11.0, 'learning_rate': 0.00013604608748078465, 'batch_size': 8.0}
    best = {'timesteps': 14.0, 'learning_rate': 0.0020131995908228796, 'batch_size': 11.0, 'l2_reg': 0.00016375804906962484}
    best = {'timesteps': 7.0, 'learning_rate': 0.00044508575407987034, 'dropouts': 'mid', 'batch_size': 5.0, 'l2_reg': 0.0034300636132326367}
    best = {'timesteps': 7.0, 'learning_rate': 0.001, 'dropouts': 'mid', 'batch_size': 5.0, 'l2_reg': 0.0034300636132326367, 'optimizer':"RMSProp"}
    best = {'optimizer': 'Adam', 'learning_rate': 0.0006030677214875642, 'batch_size': 9.0, 'timesteps': 15.0, 'dropouts': 'down', 'l2_reg': 0.00012201080729945043}
    best = {'optimizer': 'RMSProp', 'learning_rate': 0.001256172795232224, 'batch_size': 7.0, 'timesteps': 6.0, 'dropouts': 'low', 'l2_reg': 0.00015871905806080338}


    hparams = hparamsToDict( hparamsToArray( best ) )
    print( "Best params {}".format( hparams ) )

    images, y = loadData(args.dirs)
    input_dim = images[0].shape

    vals = []
    count = 5
    verbose = 0 if (count > 1) else 1
    for i in range(count):
        val = fitLSTM( input_dim, images, y, verbose=0, **hparams )
        vals.append(val)

    if count > 1:
        print( "Validation accuracy {} {} ({})".format( np.mean(vals), np.std(vals), vals ) )

if __name__ == "__main__":
    import argparse
    import malpiOptions

    parser = argparse.ArgumentParser(description='Hyperparameter Optimizer.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--number', type=int, default=100, help='Number of test runs')
    parser.add_argument('--prev', default=None, help='Skopt pickle file with data from a previous run')

    malpiOptions.addMalpiOptions( parser )
    args = parser.parse_args()
    malpiOptions.preprocessOptions(args)

    K.set_learning_phase(True)
    setCPUCores( 4 )

    if args.test_only:
        runParamTests(args)
        exit()

    logfile = "skopt_current.txt"
    holder = Holder( args.dirs, log_to=logfile )

    max_batch = 128

    space  = [
              Real(10**-10, 10**-3, "log-uniform", name='l2_reg'),
              Categorical(["low","mid","high","up","down"], name='dropouts'),
              Real(10**-9, 10**-1, "log-uniform", name='learning_rate'),
              Integer(5, max_batch, name='batch_size'),
              Categorical(["RMSProp", "Adagrad", "Adadelta", "Adam"], name='optimizer'),
              ]

    with open( logfile, 'a' ) as f:
        f.write( "#{} {} {}\n".format( "FC", "DK", datetime.datetime.now() ) )

    x0 = None
    y0 = None
    if args.prev is not None:
        x0, y0 = previousRuns(args.prev)

    res_gp = gp_minimize(holder, space, n_calls=args.number, n_random_starts=0, x0=x0, y0=y0)
 
    print( "Best: {}".format( res_gp.fun ) )
    print("""Best parameters:
    - l2_reg=%.6f
    - dropout=%s
    - learning_rate=%.6f
    - batch_size=%d
    - optimizer=%s""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3], res_gp.x[4]))

    n = datetime.datetime.now()
    fname = n.strftime('hparam_skopt_%Y%m%d_%H%M%S.pkl')
    dump( res_gp, fname, store_objective=False )
