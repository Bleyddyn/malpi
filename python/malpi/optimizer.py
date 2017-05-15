import numpy as np

class Optimizer(object):
    """ TODO: Add support for Adam: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
    """
    def __init__( self, optim_type, model, learning_rate = 0.01, decay_rate=0.99, epsilon=1e-8, upd_frequency=1 ):
        supported = ["sgd","rmsprop"]
        if optim_type not in supported:
            print "Invalid optimizer type: " % (optim_type,)
            print "Supported types: %s" % (str(supported),)
            return

        self.model = model
        self.optim_type = optim_type
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = { k : np.zeros_like(v) for k,v in self.model.params.iteritems() }
        self.upd_frequency = upd_frequency
        self.upd_count = 0
        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.params.iteritems() } # update buffers that add up gradients over a batch

    def update( self, grads, check_ratio = False ):
        for k,v in grads.iteritems():
            self.grad_buffer[k] += v
        self.upd_count += 1

        if self.upd_count % self.upd_frequency == 0:
            self.upd_count = 0
            if self.optim_type == "sgd":
                for k,v in self.model.params.iteritems():
                    g = self.grad_buffer[k] # gradient
                    self.model.params[k] -= self.learning_rate * g
            elif self.optim_type == "rmsprop":
                for k,v in self.model.params.iteritems():
                    g = self.grad_buffer[k] # gradient
                    #self._stats(g, "grad["+k+"] " )
                    self.cache[k] = self.decay_rate * self.cache[k] + (1.0 - self.decay_rate) * g**2
                    if k == "W5":
                        if check_ratio:
                            update_scale = np.linalg.norm( self.learning_rate * g / (np.sqrt(self.cache[k]) + self.epsilon) )
                            param_scale = np.linalg.norm(self.model.params[k].ravel())
                            if param_scale != 0.0:
                                ratio = update_scale / param_scale
                                if abs(ratio - 1e-3) >  0.01:
                                    print "Update ratio for %s: %f" % ( k, ratio) # should be ~1e-3
                            self._stats(self.model.params[k],"param["+k+"] " )
                            self._stats(g,"grads["+k+"] " )
                            self._stats(self.cache[k],"cache["+k+"] " )
                            diff = (self.learning_rate * g) / (np.sqrt(self.cache[k]) + self.epsilon)
                            self._stats(diff, "diffs["+k+"] " )
                            print

                    self.model.params[k] -= (self.learning_rate * g) / (np.sqrt(self.cache[k]) + self.epsilon)

            self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.params.iteritems() } # update buffers that add up gradients over a batch

    def decay_learning_rate(self, lr_decay):
        self.learning_rate *= lr_decay
         
    def _stats(self, arr, msg=""):
        mi = np.min(arr)
        ma = np.max(arr)
        av = np.mean(arr)
        std = np.std(arr)
        arr_abs = np.abs(arr)
        mi_abs = np.min(arr_abs)
        ma_abs = np.max(arr_abs)
        print "%sMin/Max/Mean/Stdev abs(Min/Max): %g/%g/%g/%g %g/%g" % (msg,mi,ma,av,std,mi_abs,ma_abs)

    def describe(self):
        print "Optimizer %s; lr=%f, dr=%f, ep=%e" % (self.optim_type, self.learning_rate, self.decay_rate, self.epsilon)
        if self.optim_type == "rmsprop":
            for k,v in self.model.params.iteritems():
                self._stats( self.cache[k], msg=("   "+k+" cache ") )

#def rmsprop(x, dx, config=None):
#  if config is None: config = {}
#  config.setdefault('learning_rate', 1e-2)
#  config.setdefault('decay_rate', 0.99)
#  config.setdefault('epsilon', 1e-8)
#  config.setdefault('cache', np.zeros_like(x))
#
#  next_x = None
#  #############################################################################
#  # TODO: Implement the RMSprop update formula, storing the next value of x   #
#  # in the next_x variable. Don't forget to update cache value stored in      #  
#  # config['cache'].                                                          #
#  #############################################################################
#  config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx**2
#  next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
#  #############################################################################
#  #                             END OF YOUR CODE                              #
#  #############################################################################
#
#  return next_x, config
