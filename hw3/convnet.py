from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import regularizers

from support import new_conv_layer, new_fc_layer, flatten_layer


class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10, weight_regularizer=None):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes
        self.weight_regularizer = weight_regularizer

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
          regularize_fc : whether or not regularize fully connected layers
        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            # it will initialize if weights are not initialized and reuse otherwise
            logits = self.__forward_pass(x)
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits


    # x : input image
    def __forward_pass(self, x):
        fc_size1 = 384
        fc_size2 = 192

        # convolutional layers
        with tf.variable_scope('conv1'):
            layer1, weights1 = new_conv_layer(x, name="conv1", num_input_channels=3, num_filters=64, filter_size=5, ac_fun=tf.nn.relu,
                                    pool_ksize=[1, 3, 3, 1])
        with tf.variable_scope('conv2'):
            layer2, weights2 = new_conv_layer(input=layer1, name="conv2", num_input_channels=64, num_filters=64, filter_size=5,
                                    ac_fun=tf.nn.relu, pool_ksize=[1, 3, 3, 1])
        with tf.name_scope('flatten'):
            layer3, num_features = flatten_layer(layer2)
        # fully connected layers
        with tf.variable_scope('fc1'):
            layer4, weights4 = new_fc_layer(input=layer3, name="fc1", num_inputs=num_features, num_outputs=fc_size1)
        # print(layer4)
        with tf.variable_scope('fc2'):
            layer5, weights5 = new_fc_layer(input=layer4, name="fc2", num_inputs=fc_size1, num_outputs=fc_size2)
        with tf.variable_scope('fc3'):
            logits, weights6 = new_fc_layer(input=layer5, name="fc3", num_inputs=fc_size2, num_outputs=self.n_classes, ac_fun=lambda x:x)


        # add histograms
        tf.histogram_summary(weights1.name, weights1)
        tf.histogram_summary(weights2.name, weights2)

        # weights we want to regularize
        self.regul_weights =[weights4, weights5, weights6]
        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        # # adding regularization
        if self.weight_regularizer is not None:
            loss += tf.add_n([self.weight_regularizer(W) for W in self.regul_weights])
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
