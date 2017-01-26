import numpy as np
import pickle

#from malpi.layers import *
from malpi.rnn_layers import *
#from malpi.fast_layers import *
#from malpi.layer_utils import *


class MalpiLSTM(object):
  """
  A simple LSTM for use with Malpi
  """
  
  def __init__(self, input_dim, hidden_dim, output_dim,
               weight_scale=1e-3, reg=0.0,
               dtype=np.float32, verbose=True):
    """
    Initialize a new network.
    
    Inputs:
    - hidden_dim: size of the hidden layer.
    - input_dim: size of input data.
    - output_dim: size of output data, in our case the number of actions.
    - weight_scale: Default scalar giving standard deviation for random initialization of weights.
    - reg: Scalar giving L2 regularization strength.
    - dtype: numpy datatype to use for computation.
    """

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.output_dim = output_dim

    # For use by describe() or higher level code
    self.hyper_parameters = {}
    self.name = ""
    self.validation_accuracy = 0.0

    self.params['Wx'] = weight_scale * np.random.randn(input_dim, 4*hidden_dim)
    self.params['Wh'] = weight_scale * np.random.randn(hidden_dim, 4*hidden_dim)
    self.params['b'] = weight_scale * np.random.randn(4*hidden_dim)
    self.params['Wo'] = weight_scale * np.random.randn(hidden_dim,output_dim)
    self.params['bo'] = weight_scale * np.random.randn(output_dim)
    self.prev_h = np.zeros((1,hidden_dim))
    self.prev_c = np.zeros((1,hidden_dim))

    self.params['bo'][0] += 1

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient.
    
    Input: Nxinput_dim array of inputs (flattened output from final cnn layer, for example)
    Output: Probabilities over number of actions/outputs
    """
    
    scores = None
    if y is None:
        mode = 'test'
    else:
        mode = 'train'

    self.prev_h, self.prev_c, cache = lstm_step_forward(X, self.prev_h, self.prev_c, self.params['Wx'], self.params['Wh'], self.params['b'])
    actions = np.dot( self.prev_h, self.params['Wo'] ) + self.params['bo']
    scores = self.softmax(actions)

    if y is None:
      return scores
    
# Most of this is wrong...
    loss, grads = 0, {}

    data_loss, dx = softmax_loss( scores, y )

    dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward( dnext_h, dnext_c, cache )

    reg_loss += 0.5 * self.reg * np.sum( self.params['Wx'] * self.params['Wx'] )
    reg_loss += 0.5 * self.reg * np.sum( self.params['Wh'] * self.params['Wh'] )
    reg_loss += 0.5 * self.reg * np.sum( self.params['Wo'] * self.params['Wo'] )

    loss = data_loss + reg_loss

    return loss, grads
  
  
  def softmax(self,x):
      probs = np.exp(x - np.max(x))
      probs /= np.sum(probs )
      return probs

  def save(self, filename):
    with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

  def describe(self):
    """
    Describe the network
    """
    
    print self.name
    print "Validation accuracy: %f" % self.validation_accuracy
    print "Hyperparameters: %s" % str(self.hyper_parameters)
    print "Input dimensions: %s" % str(self.input_dim)

    total_num_param = 0

    cnt = np.prod(self.params['Wx'].shape)
    total_num_param += cnt
    print "Wx: " + self.params['Wx'].shape + " #: " + cnt
    cnt = np.prod(self.params['Wh'].shape)
    total_num_param += cnt
    print "Wh: " + self.params['Wh'].shape + " #: " + cnt
    cnt = np.prod(self.params['b'].shape)
    total_num_param += cnt
    print "b: " + self.params['b'].shape + " #: " + cnt
    cnt = np.prod(self.params['Wo'].shape)
    total_num_param += cnt
    print "Wo: " + self.params['Wo'].shape + " #: " + cnt

    print "Total # params: " + "{:,}".format(total_num_param)

def load_malpi_lstm(filename, verbose=True):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except IOError as err:
        if verbose:
            print("IO error: {0}".format(err))
        return None

