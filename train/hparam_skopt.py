import pickle
import datetime
from skopt.space import Real, Integer, Categorical, Dimension, Identity
from skopt.utils import use_named_args
from skopt import gp_minimize

from drive_train import *

class Constant(Dimension):
    def __init__(self, value, prior=None, transform=None, name=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        * `value` [float or integer]:
            Constant value.

        * `prior` [list, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.

        * `transform` ["identity", default="identity"] :
            - "identity", the transformed space is the same as the original
              space.

        * `name` [str or None]:
            Name associated with dimension, e.g., "colors".
        """
        self.value = value
        self.name = name

        if transform is None:
            transform = "identity"
        self.transform_ = "identity"
        if transform not in ["identity"]:
            raise ValueError("Expected transform to be 'identity' or None got {}".format(transform))
        self.transformer = Identity()
        self.prior = prior

        self.prior_ = prior

    def __eq__(self, other):
        return (type(self) is type(other) and self.value == other.value)

    def __repr__(self):
        return "Constant(value={})".format(self.value)

    def rvs(self, n_samples=None, random_state=None):
        if n_samples is None:
            return self.value
        else:
            return [self.value] * n_samples

    @property
    def transformed_size(self):
        return 1

    @property
    def bounds(self):
        return [self.value]

    def __contains__(self, point):
        return point == self.value

    @property
    def transformed_bounds(self):
        return (0.0, 1.0)

    def distance(self, a, b):
        """Compute distance between category `a` and `b`.

        As categories have no order the distance between two points is one
        if a != b and zero otherwise.

        Parameters
        ----------
        * `a` [category]
            First category.

        * `b` [category]
            Second category.
        """
        return 1 if a != b else 0

class Holder:
    def __init__(self, input_dim, images, y, isFC=False):
        self.input_dim = input_dim
        self.images = images
        self.y = y
        self.vals = []
        self.run = 0
        self.isFC = isFC

#    @use_named_args(space)
    def __call__(self, args):
        print( args )
        if self.isFC:
            l2_reg, dropouts, learning_rate, batch_size, optimizer = args
        else:
            l2_reg, dropouts, learning_rate, batch_size, optimizer, timesteps = args
        arg_dict = { 'l2_reg': l2_reg,
                     'dropouts': dropouts,
                     'learning_rate': learning_rate,
                     'batch_size': batch_size,
                     'optimizer': optimizer }
        if not self.isFC:
            arg_dict['timesteps'] = timesteps

        self.run += 1
        hparams = hparamsToDict( hparamsToArray( arg_dict ) )
        print( "Run {}".format( self.run ) )
        print( "   Args {}".format( args ) )
        print( "   Hparams {}".format( hparams ) )
        if self.isFC:
            val, his, model = fitFC( self.input_dim, self.images, self.y, verbose=0, **hparams )
        else:
            val, his, model = fitLSTM( self.input_dim, self.images, self.y, verbose=0, **hparams )
        self.vals.append(val)
        print( "   Val acc {}".format( val ) )
        #with open( 'hparam_current.txt', 'a' ) as f:
        #    f.write( "{}\n".format( val ) )
        #ret = { 'loss': 1.0 - val, 'status': STATUS_OK, 'history':pickle.dumps(his.history), 'val':val }
        return 1.0 - val


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

    """
    space  = [
              Constant(0, 0, name='timesteps'),
              Real(10**-10, 10**-3, "log-uniform", name='l2_reg'),
              Categorical(["low","mid","high","up","down"], name='dropouts'),
              Real(10**-9, 10**-4, "log-uniform", name='learning_rate'),
              Constant(0.2, 0.2, name='validation_split'),
              Integer(5, max_batch, name='batch_size'),
              Categorical(["RMSProp", "Adagrad", "Adadelta", "Adam"], name='optimizer'),
              Constant(40, 40, name='epochs')
              ]
    """
    space  = [
              Real(10**-10, 10**-3, "log-uniform", name='l2_reg'),
              Categorical(["low","mid","high","up","down"], name='dropouts'),
              Real(10**-9, 10**-4, "log-uniform", name='learning_rate'),
              Integer(5, max_batch, name='batch_size'),
              Categorical(["RMSProp", "Adagrad", "Adadelta", "Adam"], name='optimizer'),
              ]

    if not args.fc:
        space.append( Integer(5,20,name='timesteps') )


#    with open( 'gpopt_current.txt', 'a' ) as f:
#        f.write( "#{} {} {}\n".format( ("FC" if args.fc else "RNN"), ("DK" if args.dk else "DM"), datetime.datetime.now() ) )

    res_gp = gp_minimize(holder, space, n_calls=10)
 
    print( "Best: {}".format( res_gp.fun ) )
    print("""Best parameters:
    - l2_reg=%.6f
    - dropout=%s
    - learning_rate=%.6f
    - batch_size=%d
    - optimizer=%s""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3], res_gp.x[4]))

    n = datetime.datetime.now()
    fname = n.strftime('hparam_gpopt_%Y%m%d_%H%M%S.pkl')
    with open(fname,'w') as f:
        pickle.dump( res_gp, f, pickle.HIGHEST_PROTOCOL)
