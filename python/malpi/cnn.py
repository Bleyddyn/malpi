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
               num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3,
               reg=0.0,
               dtype=np.float32, dropout=0.0):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
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
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    self.dropout_param = {'mode': 'train', 'p': dropout}

    stride = self.conv_param['stride']
    pad = self.conv_param['pad']
    ph = self.pool_param['pool_height']
    pw = self.pool_param['pool_width']
    ps = self.pool_param['stride']
    W2 = (input_dim[1]-filter_size+2*pad)/stride+1
# take into account the pooling layer
    W2 = (W2 - pw)/ps + 1
    H2 = (input_dim[2]-filter_size+2*pad)/stride+1
    H2 = (H2 - ph)/ps + 1

    layer_num = 1
    output_dim = input_dim
    for layer in layers:

        layer_num_str = str(layer_num)
        params = layer_params[layer_num-1]
        wscale = weight_scale
        if 'weight_scale' in params:
            wscale = params['weight_scale']

        if layer.startswith(("conv-","Conv-")):
            num_filters = int( layer.split("-",1)[1] )
            w, h = self.get_conv_filter_sizes(params, default=filter_size)
            pad, stride = self.get_conv_stride(params, w, h, output_dim[-2], output_dim[-1])
            self.params['W'+layer_num_str] = wscale * np.random.randn(num_filters,output_dim[0],w,h)
# Should be using: w = np.random.randn(n) * sqrt(2.0/n)
# for Relu neurons Where n is the number of inputs
            self.params['b'+layer_num_str] = np.zeros(num_filters)
            w = (output_dim[-2]-w+2*pad)/stride+1
            h = (output_dim[-1]-h+2*pad)/stride+1
            output_dim = np.array([num_filters,w,h])
            layer_num += 1
            print "Conv %d (%d/%d)" % (num_filters,w,h)
        elif layer.startswith(("fc-","FC-")):
            hidden = int( layer.split("-",1)[1] )
            self.params['W'+layer_num_str] = wscale * np.random.randn(np.prod(output_dim),hidden)
            self.params['b'+layer_num_str] = np.zeros(hidden)
            output_dim = np.array([hidden])
            layer_num += 1
            print "FC %d" % (hidden,)
        elif layer.startswith(("maxpool","Maxpool")):
            pool_w = params['pool_width']
            pool_h = params['pool_height']
            pool_str = params['pool_stride']
            w = (output_dim[-2] - pool_w) / pool_str + 1
            h = (output_dim[-1] - pool_h) / pool_str + 1
            output_dim[-2] = w
            output_dim[-1] = h
            print "pool"
        print "  ", params
        print "  ", output_dim

    print self.params.keys()
#    self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
#    self.params['b1'] = np.zeros(num_filters)
#    self.params['W2'] = weight_scale * np.random.randn(num_filters,num_filters,filter_size,filter_size)
#    self.params['b2'] = np.zeros(num_filters)
#
#    self.params['W3'] = weight_scale * np.random.randn( num_filters*W2*H2, hidden_dim )
#    self.params['b3'] = np.zeros(hidden_dim)
#    self.params['W4'] = weight_scale * np.random.randn( hidden_dim, num_classes )
#    self.params['b4'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
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
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    self.dropout_param['mode'] = 'test' if y is None else 'train'
    out, cache1 = conv_relu_forward(X,W1,b1, self.conv_param)
    out, cache2 = conv_relu_pool_forward(out,W2,b2, self.conv_param, self.pool_param)
    out, cache3 = affine_relu_forward(out, W3, b3)
    if self.use_dropout:
        out, dcache = dropout_forward(out, self.dropout_param)
    scores, cache4 = affine_forward(out, W4, b4)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dx = softmax_loss( scores, y )

    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4))
    loss = data_loss + reg_loss

    dx, dw, db = affine_backward(dx, cache4)
    dw += self.reg * W4
    grads['W4'] = dw
    grads['b4'] = np.reshape(db, self.params['b4'].shape)
    
    if self.use_dropout:
        dx = dropout_backward(dx, dcache)

    dx, dw, db = affine_relu_backward(dx, cache3)
    dw += self.reg * W3
    grads['W3'] = dw
    grads['b3'] = np.reshape(db, self.params['b3'].shape)

    dx, dw, db = conv_relu_pool_backward(dx, cache2)
    dw += self.reg * W2
    grads['W2'] = dw
    grads['b2'] = np.reshape(db, self.params['b2'].shape)

    dx, dw, db = conv_relu_backward(dx, cache1)
    dw += self.reg * W1
    grads['W1'] = dw
    grads['b1'] = np.reshape(db, self.params['b1'].shape)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
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
    if 'pad' in params:
        pad = params['pad']
    else:
        pad = int( (fw - 1) / 2 )
    return (pad, stride)

  def save(self, filename):
    with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def load_malpi(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

