"""
This module implements various layers for the network.
You should fill in code into indicated sections.
"""
import numpy as np
from support import L2
import math

class Layer(object):
  """
  Base class for all layers classes.

  """
  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer:
          input_size - input dimension;
          output_size - output dimension;
          weight_decay - L2-regularization parameter for the weights;
          weight_scale - scale of normal distrubtion to initialize weights.

    """

    self.layer_params = layer_params
    self.layer_params.setdefault('weight_decay', 0.0)
    self.layer_params.setdefault('weight_scale', 0.0001)

    self.params = {'w': None, 'b': None}
    self.grads = {'w': None, 'b': None}

    self.train_mode = False


  def initialize(self):
    """
    Cleans cache. Cache stores intermediate variables needed for backward computation.

    """
    ########################################################################################
    # Initialize weights self.params['w'] using normal distribution with mean = 0 and      #
    # std = self.layer_params['weight_scale'].                                             #
    #                                                                                      #
    # Initialize biases self.params['b'] with 0.                                           #
    ########################################################################################
    self.params['w'] = np.random.normal(size=(self.layer_params['input_size'], self.layer_params['output_size']),
                                        loc=0, scale=self.layer_params['weight_scale'])
    self.params['b'] = np.zeros(shape=(self.layer_params['output_size'], ), dtype="float32")
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    self.cache = None

  def layer_loss(self):
    """
    Returns the loss of the layer parameters for the regularization term of full network loss.

    Returns:
      loss: Loss of the layer parameters for the regularization term of full network loss.

    """

    ########################################################################################
    # Compute the loss of the layer which responsible for L2 regularization term. Store it #
    # in loss variable.                                                                    #
    ########################################################################################
    loss = L2(self.params['w'])
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return loss

  def set_train_mode(self):
    """
    Sets train mode for the layer.

    """
    self.train_mode = True

  def set_test_mode(self):
    """
    Sets test mode for the layer.

    """
    self.train_mode = False

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # Implement forward pass for LinearLayer. Store output of the layer in out variable.    #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    s = np.dot(x, self.params['w']) + self.params['b']

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    if self.cache is None:
        self.cache = {}
    self.cache['z_prev'] = x
    self.cache['s_cur'] = s

    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return self.act(s)

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradient of the output with respect to the input of the layer.

    """
    de = self.der_act(self.cache['s_cur'])
    dx = np.dot(dout* de, self.params['w'].T)
    self.grads['w'] = np.dot(self.cache['z_prev'].T, dout) + self.layer_params['weight_decay'] * self.params['w']
    self.grads['b'] = np.sum(dout, axis=0, keepdims=True)

    return dx


class LinearLayer(Layer):
  """
  Linear layer.

  """
  def act(self, x):
      return x

  def der_act(self, x):
      return 1



class ReLULayer(Layer):
  """
  ReLU activation layer.

  """
  def act(self, x ):
      return np.maximum(x, 0)

  def der_act(self, x):
      return x > 0


class SigmoidLayer(Layer):
  """
  Sigmoid activation layer.

  """
  def act(self, x):
    return 1.0 / (1.0 + math.exp(-x))


  def der_act(self, x):
    act = self.act(x)
    return act* (1- act)


class TanhLayer(Layer):
  """
  Tanh activation layer.

  """
  def act(self, x):
    return np.tanh(x)

  def der_act(self, x):
    return 1 - self.act(self.act(x))


class ELULayer(Layer):
  """
  ELU activation layer.

  """

  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer:
          alpha - alpha parameter;

    """
    self.layer_params = layer_params
    self.layer_params.setdefault('alpha', 1.0)
    self.train_mode = False

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for ELULayer. Store output of the layer in out variable.      #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ELULayer. Store gradient of the loss with respect to     #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class SoftMaxLayer(Layer):
  """
  Softmax activation layer.

  """

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for SoftMaxLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SoftMaxLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################s
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx
