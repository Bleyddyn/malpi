from hyperopt import fmin, tpe, hp
import hyperopt.pyll.stochastic
from drive_train import *

class Holder:
    def __init__(self, input_dim, images, y):
        self.input_dim = input_dim
        self.images = images
        self.y = y
        self.vals = []
        self.run = 0

    def __call__(self, args):
        self.run += 1
        hparams = hparamsToDict( hparamsToArray( args ) )
        print( "Run {}".format( self.run ) )
        print( "   Args {}".format( args ) )
        print( "   Hparams {}".format( hparams ) )
        val = fitLSTM( self.input_dim, self.images, self.y, verbose=0, **hparams )
        self.vals.append(val)
        print( "   Val acc {}".format( val ) )
        return 1.0 - val

#def fitLSTM( input_dim, images, y, verbose=1, epochs=40, timesteps=10, l2_reg=0.005, dropouts=[0.25,0.25,0.25,0.25,0.25],
#             learning_rate=0.003, validation_split=0.15, batch_size=5, optimizer="RMSprop" ):

def runParamTests(args):
    best = {'timesteps': 7.0, 'learning_rate': 0.001306236693845287, 'batch_size': 8.0}
    best = {'timesteps': 11.0, 'learning_rate': 0.00013604608748078465, 'batch_size': 8.0}
    best = {'timesteps': 14.0, 'learning_rate': 0.0020131995908228796, 'batch_size': 11.0, 'l2_reg': 0.00016375804906962484}

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

    holder = Holder(input_dim, images, y)

#timesteps = 10
#l2_reg = 0.005
#dropouts = [0.25,0.25,0.25,0.25,0.25]
#learning_rate = 0.003
#optimizer = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.005)
#validation_split=0.15
#epochs=40
#batch_size=5

    space = { 'learning_rate': hp.loguniform('learning_rate', -9, -4 ),
              'l2_reg': hp.loguniform('l2_reg', -10, -3 ),
              'timesteps': hp.quniform('timesteps', 5, 20, 1 ),
              'batch_size': hp.quniform('batch_size', 5, 20, 1),
              'dropouts': hp.choice('dropouts', ["low","mid","high","up","down"]) }

#space = hp.choice('a',
#     [
#         ('case 1', 1 + hp.lognormal('c1', 0, 1)),
#         ('case 2', hp.uniform('c2', -10, 10))
#     ])

#for i in range(10):
#    print( "Sample: {}".format( hyperopt.pyll.stochastic.sample(space) ) )
# {'timesteps': 18.0, 'batchsize': 16.566825420405156}

    best = fmin(fn=holder, space=space, algo=tpe.suggest, max_evals=100)
    print( "Best: {}".format( best ) )
    plt.plot(holder.vals)
    plt.show()

