from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from support import new_conv_layer, new_fc_layer, flatten_layer



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse=False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('Siamese', reuse=reuse):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            ########################
            logits = self.__forward_pass(x, reuse)
            l2_out = tf.nn.l2_normalize(logits, dim=1)
            ########################

        return l2_out


    def __forward_pass(self, x, reuse):
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
            logits, weights5 = new_fc_layer(input=layer4, name="fc2", num_inputs=fc_size1, num_outputs=fc_size2)

        # add histograms
        if not reuse:
            tf.histogram_summary(weights1.name, weights1)
            tf.histogram_summary(weights2.name, weights2)

        return logits


    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        D = (tf.reduce_sum((channel_1 - channel_2)**2, reduction_indices=1))**0.5
        zeros = tf.fill(tf.shape(D), 0.0)
        # loss = 0.5*(label*(D**2.) + (1-label) * (tf.reduce_max([zeros, margin - D], reduction_indices=0))**2)
        loss = label*(D**2) + (1-label) * (tf.reduce_max([zeros, margin - D**2], 0))
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
