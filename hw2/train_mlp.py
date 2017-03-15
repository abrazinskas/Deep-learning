"""
This module implements training and evaluation of a multi-layer perceptron.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
from cifar10_utils import get_cifar10
from support import accuracy_loss_curve, get_run_var
import numpy as np

from mlp import MLP

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.00
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'
### --- END default constants---

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#initializers
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#sharing-variables
WEIGHT_INITIALIZATION_DICT = {'xavier': lambda dummy: initializers.xavier_initializer(),  # Xavier initialisation
                              'normal': lambda sd: tf.random_normal_initializer(stddev=sd),  # Initialization from a standard normal
                              'uniform': lambda lim: tf.random_uniform_initializer(minval=-lim, maxval=lim),  # Initialization from a uniform distribution
                              }

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#regularizers
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#sharing-variables
WEIGHT_REGULARIZER_DICT = {'none': lambda val: None,  # No regularization
                           'l1': lambda val: regularizers.l1_regularizer(val),  # L1 regularization
                           'l2': lambda val: regularizers.l2_regularizer(val)  # L2 regularization
                           }

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#activation-functions
ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.nn.tanh,  # Tanh
                   'sigmoid': tf.nn.sigmoid}  # Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer,  # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

FLAGS = None

CHECK_STATISTICS_STEPS = 100  # after how many steps should we check statistics such as loss and accuracy

def train():
    """
    Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
    as you did in the practical 1. This time evaluate your model on the whole test set.
    """
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    tf.set_random_seed(42)
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    params = vars(FLAGS)

    params['weight_reg'] = params['weight_reg'] if params['weight_reg_strength']>0 else 'none'
    learning_rate = params['learning_rate']
    max_steps = params['max_steps']
    batch_size = params['batch_size']
    weight_init_scale = float(params['weight_init_scale'])
    weight_init = WEIGHT_INITIALIZATION_DICT[params['weight_init']]
    weight_reg = WEIGHT_REGULARIZER_DICT[params['weight_reg']]
    weight_reg_strength = params['weight_reg_strength']
    dropout_rate = params['dropout_rate']
    activation_fn = ACTIVATION_DICT[params['activation']]
    optimizer = OPTIMIZER_DICT[params['optimizer']]
    data_dir = params['data_dir']
    log_dir = params['log_dir']

    tf.reset_default_graph()

    train_set, val_set, test_set = get_cifar10(data_dir)
    n_classes = train_set.labels.shape[1]
    m, n = train_set.images.shape

    # X_data, y_data, n_classes  = load_data(data_dir)

    mlp = MLP(n_hidden=dnn_hidden_units, n_classes=n_classes,
              activation_fn=activation_fn, dropout_rate=dropout_rate, weight_initializer=weight_init(weight_init_scale),
              weight_regularizer=weight_reg(weight_reg_strength))
    mlp = initialize_parameters(mlp, n_features=n)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=(None, n))
        y = tf.placeholder(tf.int32, shape=(None, n_classes))

    with tf.name_scope('test'):
        x_test = test_set.images
        y_test = test_set.labels

    with tf.name_scope('inference'):
        logits = mlp.inference(x)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(mlp.loss(logits, y))

    with tf.name_scope('optimize'):
        opt = optimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('Accuracy'):
        accuracy = mlp.accuracy(logits, y)
    tf.histogram_summary("logits", logits)

    # saving history ( for custom graphs)
    train_losses = []
    test_losses = []
    test_accuracies = []
    train_accuracies = []
    iterations = []

    tf.scalar_summary("loss", loss)
    merged_op = tf.merge_all_summaries()

    # Initializing the variables
    init = tf.initialize_all_variables()
    # Launch the graph
    with tf.Session() as sess:
        run_var = get_run_var(log_dir)
        summary_writer = tf.train.SummaryWriter('%s/%s' % (log_dir, run_var), graph=tf.get_default_graph())


        sess.run(init)

        # compute initial accuracy
        mlp.set_testing_mode()
        acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        mlp.set_training_mode()

        print ("before training test accuracy is %f " % acc)
        # training cycle
        for step in range(1, max_steps+1):
            x_batch, y_batch = train_set.next_batch(batch_size)
            _, loss_val, summary_str = sess.run([opt, loss, merged_op], feed_dict={x: x_batch, y: y_batch})
            summary_writer.add_summary(summary_str, step)
            if step % CHECK_STATISTICS_STEPS == 0:
                print("Step:", '%04d' % step, "1 batch loss=", "{:.9f}".format(loss_val))

                mlp.set_testing_mode()
                test_loss = sess.run(loss, feed_dict={x: x_test, y: y_test})
                test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
                train_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
                mlp.set_training_mode()

                # save
                iterations.append(step)
                train_losses.append(loss_val)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                train_accuracies.append(train_accuracy)

        # after training accuracy
        mlp.set_testing_mode()
        acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print ("after training test accuracy is %f " % acc)

        # print confusion matrix
        my_logits = sess.run(logits, feed_dict={x: x_test, y: y_test})
        predictions = tf.argmax(my_logits, 1)
        true_labels = tf.argmax(y_test, 1)
        print (get_confusion_matrix(predictions, true_labels).eval())

        # plot
        accuracy_loss_curve(train_accuracies, test_accuracies, train_losses, test_losses, iterations)


        summary_writer.flush()
    ########################
    # END OF YOUR CODE    #
    #######################

# initializes parameters for a neural networks, in principle this method can be placed to the model's class
def initialize_parameters(model, n_features):
    # Store layers weight & bias
    weights = []
    biases = []

    with tf.name_scope('parameters'):

        # add hidden layers
        prev_dim = n_features
        for i, hid_dim in enumerate(model.n_hidden):
            weight = tf.get_variable(name="h_" + str(i + 1) + "_weight", shape=(prev_dim, hid_dim),
                                     initializer=model.weight_initializer)
            bias = tf.get_variable(name="b_" + str(i + 1), shape=(hid_dim, ), initializer=tf.constant_initializer(0.0))
            weights.append(weight)
            biases.append(bias)
            prev_dim = hid_dim

        # add output layer
        weight = tf.get_variable(name="output_weight", shape=(model.n_hidden[-1], model.n_classes))
        bias = tf.get_variable(name="b_output", shape=(model.n_classes, ), initializer=tf.constant_initializer(0.0))
        weights.append(weight)
        biases.append(bias)

        model.weights = weights
        model.biases = biases

    # add histograms
    for W in model.weights:
        tf.histogram_summary(W.name, W)
    for b in model.biases:
        tf.histogram_summary(b.name, b)

    return model

def get_confusion_matrix(plabels, tlabels):
    return tf.contrib.metrics.confusion_matrix(plabels, tlabels)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main(_):
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    # Make directories if they do not exists yet
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--weight_init', type=str, default=WEIGHT_INITIALIZATION_DEFAULT,
                        help='Weight initialization type [xavier, normal, uniform].')
    parser.add_argument('--weight_init_scale', type=float, default=WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                        help='Weight initialization scale (e.g. std of a Gaussian).')
    parser.add_argument('--weight_reg', type=str, default=WEIGHT_REGULARIZER_DEFAULT,
                        help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
    parser.add_argument('--weight_reg_strength', type=float, default=WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE_DEFAULT,
                        help='Dropout rate.')
    parser.add_argument('--activation', type=str, default=ACTIVATION_DEFAULT,
                        help='Activation function [relu, elu, tanh, sigmoid].')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
