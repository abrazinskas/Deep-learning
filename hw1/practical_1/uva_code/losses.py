"""
This module implements various losses for the network.
You should fill in code into indicated sections.
"""
from support import sm, total_ce_loss, total_ce_loss_grad
import numpy as np

def HingeLoss(x, y):
  """
  Computes multi-class hinge loss and gradient of the loss with the respect to the input for multiclass SVM.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar hinge loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # Compute hinge loss on input x and y and store it in loss variable. Compute gradient  #
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  m, k = x.shape
  margin = 0.1

  cor = x[range(m), y].reshape((-1,1))
  dx = x - cor + margin

  dx = (dx > 0) *1
  dx[range(m), y] = - np.sum(dx, axis=1)
  loss = np.mean(np.sum(np.maximum(0, x - cor + margin), axis=1, keepdims=1))
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx/float(m)

def CrossEntropyLoss(x, y):
  """
  Computes multi-class cross entropy loss and gradient with the respect to the input x.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar multi-class cross entropy loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # Compute cross entropy loss on input x and y and store it in loss variable. Compute   #
  # gradient of the loss with respect to the input and store it in dx.                   #
  ########################################################################################
  m, k = x.shape
  one_hot = np.zeros((m, k))
  one_hot[range(m), y] = 1
  dx =  - one_hot / x
  loss = - np.sum(np.log(x[range(m), y]))/float(m)
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx/float(m)

def SoftMaxLoss(x, y):
  """
  Computes the loss and gradient with the respect to the input x.

  Args:
    x: Input data (matrix).
    y: Labels of data (vector).

  Returns:
    loss: Scalar softmax loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # Compute softmax loss on input x and y and store it in loss variable. Compute gradient#
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  y_hat = sm(x) # [batch x classes]
  m = x.shape[0]
  loss = - np.sum(np.log(y_hat[range(m), y]))/float(m)
  dx = y_hat
  dx[range(m), y] -= 1

  # loss, dx = CrossEntropyLoss(y_hat, y)
  # loc_dx = - y_hat
  # loc_dx = loc_dx * y_hat[range(m), y].reshape((-1,1)) # fixing for incorrect predictions
  # loc_dx[range(m), y] /= y_hat[range(m), y]
  # loc_dx[range(m), y] = loc_dx[range(m), y] - loc_dx[range(m), y]**2
  # dx = dx * loc_dx
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx/float(m)
