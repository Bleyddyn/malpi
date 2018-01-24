import pickle
import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
import matplotlib.pyplot as plt
from drive_train import *

class Holder:
    def __init__(self, input_dim, images, y, isFC=False):
        self.input_dim = input_dim
        self.images = images
        self.y = y
        self.vals = []
        self.run = 0
        self.isFC = not isFC

    def __call__(self, args):
        self.run += 1
        hparams = hparamsToDict( hparamsToArray( args ) )
        print( "Run {}".format( self.run ) )
        print( "   Args {}".format( args ) )
        print( "   Hparams {}".format( hparams ) )
        if self.isFC:
            val, his = fitFC( self.input_dim, self.images, self.y, verbose=0, **hparams )
        else:
            val, his = fitLSTM( self.input_dim, self.images, self.y, verbose=0, **hparams )
        self.vals.append(val)
        print( "   Val acc {}".format( val ) )
        ret = { 'loss': 1.0 - val, 'status': STATUS_OK, 'history':pickle.dumps(his.history), 'val':val }
        return ret


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
    args = getOptions()

    K.set_learning_phase(True)
    setCPUCores( 4 )

    if args.test_only:
        runParamTests(args)
        exit()

    images, y = loadData(args.dirs)
    input_dim = images[0].shape
    num_actions = len(y[0])
    num_samples = len(images)

    holder = Holder(input_dim, images, y, args.fc)

    if args.fc:
        max_batch = 128
    else:
        max_batch = 20

    space = { 'learning_rate': hp.loguniform('learning_rate', -9, -4 ),
              'l2_reg': hp.loguniform('l2_reg', -10, -3 ),
              'batch_size': hp.quniform('batch_size', 5, max_batch, 1),
              'dropouts': hp.choice('dropouts', ["low","mid","high","up","down"]),
              'optimizer': hp.choice('optimizer', ["RMSProp", "Adagrad", "Adadelta", "Adam"]),
              'epochs': 40 }
    if not args.fc:
        space['timesteps'] = hp.quniform('timesteps', 5, 20, 1 )


#space = hp.choice('a',
#     [
#         ('case 1', 1 + hp.lognormal('c1', 0, 1)),
#         ('case 2', hp.uniform('c2', -10, 10))
#     ])

#for i in range(10):
#    print( "Sample: {}".format( hyperopt.pyll.stochastic.sample(space) ) )
# {'timesteps': 18.0, 'batchsize': 16.566825420405156}

    trials = Trials()
    best = fmin(fn=holder, space=space, algo=tpe.suggest, max_evals=100, trials=trials )
    print( "Best: {}".format( best ) )
    print( "Val Accuracies: {}".format( holder.vals ) )
    plt.plot(holder.vals)
    plt.show()

    n = datetime.datetime.now()
    fname = n.strftime('hparam_trials_%Y%m%d_%H%M%S.pkl')
    with open(fname,'w') as f:
        pickle.dump( trials, f, pickle.HIGHEST_PROTOCOL)
