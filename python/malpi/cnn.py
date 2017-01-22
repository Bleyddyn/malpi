import numpy as np
import pickle

from malpi.layers import *
from malpi.fast_layers import *
from malpi.layer_utils import *


class MalpiConvNet(object):
  """
  A multi-layer convolutional network for experimenting with the architecture.

  e.g.:  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, layers, layer_params, input_dim=(3, 32, 32),
               weight_scale=1e-3, reg=0.0,
               dtype=np.float32, dropout=0.0, verbose=True):
    """
    Initialize a new network.
    
    Inputs:
    - layers: Array of strings describing network topology. e.g. ['conv-32','maxpool','fc-10']
    - layer_params: Array of tuples giving parameters for each layer. Must be same length as layers.
    - input_dim: Tuple (C, H, W) giving size of input data.
    - weight_scale: Default scalar giving standard deviation for random initialization of weights.
    - reg: Scalar giving L2 regularization strength.
    - dtype: numpy datatype to use for computation.


    Try: Conv-64, Conv-64, maxpool, conv-128, conv-128, maxpool, conv-256, conv-256, maxpool, conv-512, conv-512, maxpool, conv-512, conv-512, maxpool, FC-4096, FC-4096, FC-1000, softmax
    """

    if len(layers) != len(layer_params):
        print "layers and layer_params parameters must be the same length.\nUse an empty tuple for layers with no params."
        return

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_dropout = dropout > 0
    self.name = ""
    self.validation_accuracy = 0.0
    self.input_dim = input_dim
    self.hyper_parameters = {}
    self.dropout_param = {'mode': 'train', 'p': dropout}

    self.layers = layers
    self.layer_params = layer_params

    layer_num = 0
    output_dim = input_dim
    for layer in layers:

        layer_num += 1
        layer_num_str = str(layer_num)
        params = layer_params[layer_num-1]
        wscale = weight_scale
        if 'weight_scale' in params:
            wscale = params['weight_scale']

        if layer.startswith(("conv-","Conv-")):
            num_filters = int( layer.split("-",1)[1] )
            w, h = self.get_conv_filter_sizes(params)
            pad, stride = self.get_conv_stride(params, w, h, output_dim[-2], output_dim[-1])
            self.params['W'+layer_num_str] = np.sqrt( 2.0 / np.prod(output_dim)) * np.random.randn(num_filters,output_dim[0],w,h)
# TODO: Should be using: w = np.random.randn(n) * sqrt(2.0/n)
# for Relu neurons Where n is the number of inputs
            self.params['b'+layer_num_str] = np.zeros(num_filters)
            w = (output_dim[-2]-w+2*pad)/stride+1
            h = (output_dim[-1]-h+2*pad)/stride+1
            output_dim = np.array([num_filters,w,h])
            if verbose:
                print "Conv %d (%d/%d)" % (num_filters,w,h)
        elif layer.startswith(("fc-","FC-")):
            hidden = int( layer.split("-",1)[1] )
            if 'relu' in layer_params and not layer_params['relu']:
                lwscale = wscale
            else:
                lwscale = np.sqrt( 2.0 / np.prod(output_dim) )
            self.params['W'+layer_num_str] = lwscale * np.random.randn(np.prod(output_dim),hidden)
            self.params['b'+layer_num_str] = np.zeros(hidden)
            output_dim = np.array([hidden])
            if verbose:
                print "FC %d" % (hidden,)
        elif layer.startswith(("maxpool","Maxpool")):
            pool_w = params['pool_width']
            pool_h = params['pool_height']
            pool_str = params['pool_stride']
            w = (output_dim[-2] - pool_w) / pool_str + 1
            h = (output_dim[-1] - pool_h) / pool_str + 1
            output_dim[-2] = w
            output_dim[-1] = h
            if verbose:
                print "pool"
        if verbose:
            print "  params: ", params
            print "  outdim: ", output_dim

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    scores = None
    #def affine_relu_forward(x, w, b):
    #def affine_relu_backward(dout, cache):
    #def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    #def affine_bn_relu_backward(dout, cache):
    #def conv_relu_forward(x, w, b, conv_param):
    #def conv_relu_backward(dout, cache):
    #def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    #def conv_relu_pool_backward(dout, cache):
    #def conv_sbn_relu_forward(x, w, b, conv_param, sbn_param):
    #def conv_sbn_relu_backward(dout, cache):

    if y is None:
        mode = 'test'
    else:
        mode = 'train'

    layer_num = 0
    layer_caches = []
    inputs = X
    for layer in self.layers:
        layer_num += 1
        layer_num_str = str(layer_num)
        params = self.layer_params[layer_num-1]

        if layer.startswith(("conv-","Conv-")):
            W = self.params['W'+layer_num_str]
            b = self.params['b'+layer_num_str]
            layer_params = self.layer_params[layer_num-1]
            inputs, cache = conv_relu_forward( inputs, W, b, layer_params, mode=mode )
            layer_caches.append(cache)

        elif layer.startswith(("fc-","FC-")):
            W = self.params['W'+layer_num_str]
            b = self.params['b'+layer_num_str]
            layer_params = self.layer_params[layer_num-1]
            if 'relu' in layer_params and not layer_params['relu']:
                inputs, cache = affine_forward( inputs, W, b )
            else:
                inputs, cache = affine_relu_forward( inputs, W, b )
            layer_caches.append(cache)

        elif layer.startswith(("maxpool","Maxpool")):
            layer_params = self.layer_params[layer_num-1]
            inputs, pool_cache = max_pool_forward_fast( inputs, layer_params, mode=mode )
            layer_caches.append(pool_cache)

    scores = inputs

    if y is None:
      return scores
    
    loss, grads = 0, {}

    data_loss, dx = softmax_loss( scores, y )

    reg_loss = 0

    layer_num = len(self.layers)+1
    for layer in reversed(self.layers):
        layer_num -= 1
        layer_num_str = str(layer_num)
        params = self.layer_params[layer_num-1]

        if layer.startswith(("conv-","Conv-")):
            W = self.params['W'+layer_num_str]
            b = self.params['b'+layer_num_str]
            layer_params = self.layer_params[layer_num-1]
            cache = layer_caches[layer_num-1]
            reg_loss += 0.5 * self.reg * np.sum(W*W)
            dx, dw, db = conv_relu_backward( dx, cache )
            dw += self.reg * W
            grads['W'+layer_num_str] = dw
            grads['b'+layer_num_str] = np.reshape(db, b.shape)

        elif layer.startswith(("fc-","FC-")):
            W = self.params['W'+layer_num_str]
            b = self.params['b'+layer_num_str]
            layer_params = self.layer_params[layer_num-1]
            cache = layer_caches[layer_num-1]
            reg_loss += 0.5 * self.reg * np.sum(W*W)
            if 'relu' in layer_params and not layer_params['relu']:
                dx, dw,db = affine_backward( dx, cache )
            else:
                dx, dw, db = affine_relu_backward( dx, cache )
            dw += self.reg * W
            grads['W'+layer_num_str] = dw
            grads['b'+layer_num_str] = np.reshape(db, b.shape)

        elif layer.startswith(("maxpool","Maxpool")):
            cache = layer_caches[layer_num-1]
            dx = max_pool_backward_fast( dx, cache )

    loss = data_loss + reg_loss

    return loss, grads
  
  
  def get_conv_filter_sizes(self, params, default=3):
    filter_width = default
    filter_height = default
    if 'filter_size' in params:
        filter_width = params['filter_size']
        filter_height = filter_width
    if 'filter_width' in params:
        filter_width = params['filter_width']
    if 'filter_height' in params:
        filter_height = params['filter_height']
    return (filter_width,filter_height)

  def get_conv_stride(self, params, fw, fh, iw, ih):
    stride = 1
    if 'stride' in params:
        stride = params['stride']
    else:
        params['stride'] = stride
    if 'pad' in params:
        pad = params['pad']
    else:
        pad = int( (fw - 1) / 2 )
        params['pad'] = pad
    return (pad, stride)

  def save(self, filename):
    with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

  def describe(self):
    """
    Describe the network
    """
    
    layer_num = 0
    total_num_param = 0
    print self.name
    print "Validation accuracy: %f" % self.validation_accuracy
    print "Hyperparameters: %s" % str(self.hyper_parameters)
    print "Input dimensions: %s" % str(self.input_dim)
    for layer in self.layers:
        layer_num += 1
        layer_num_str = str(layer_num)
        params = self.layer_params[layer_num-1]

        if layer.startswith(("conv-","Conv-")):
            W = self.params['W'+layer_num_str]
            b = self.params['b'+layer_num_str]
            cnt = np.prod( W.shape ) + np.prod( b.shape )
            total_num_param += cnt
            print layer
            print "    " + str(params)
            print "    W: " + str(W.shape) + " b: " + str(b.shape) + " #: " + str(cnt)

        elif layer.startswith(("fc-","FC-")):
            W = self.params['W'+layer_num_str]
            b = self.params['b'+layer_num_str]
            cnt = np.prod( W.shape ) + np.prod( b.shape )
            total_num_param += cnt
            print layer
            print "    " + str(params)
            print "    W: " + str(W.shape) + " b: " + str(b.shape) + " #: " + str(cnt)

        elif layer.startswith(("maxpool","Maxpool")):
            print layer
            print "    " + str(params)
    print "Total # params: " + "{:,}".format(total_num_param)

def load_malpi(filename, verbose=True):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except IOError as err:
        if verbose:
            print("IO error: {0}".format(err))
        return None

