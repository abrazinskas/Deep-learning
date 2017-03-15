from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from cifar10_utils import get_cifar10
from vgg import VGG
from vgg_support import load_pretrained_VGG16_pool5
from support import get_run_var, save_accuracy_curve_in_a_file
from tensorflow.contrib.layers import regularizers

import tensorflow as tf
import numpy as np

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0
REGULARIZATION_STRENGTH_DEFAULT = 0.

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer,  # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

def train_step(loss, optimizer_type="adam", learning_rate=1e-4):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    with tf.name_scope('optimize'):
        optimizer = OPTIMIZER_DICT[optimizer_type]
        train_op = optimizer(learning_rate=learning_rate).minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op


def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    params = vars(FLAGS)
    max_steps = params['max_steps']
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    log_dir = params['log_dir']
    checkpoint_freq = params['checkpoint_freq']
    checkpoint_dir = params['checkpoint_dir']
    train_eval_freq = params['print_freq']
    test_eval_freq = params['eval_freq']
    learning_rate = params['learning_rate']
    refine_after_k = params['refine_after_k']
    reg_strength = params['reg_strength']
    regularizer = regularizers.l2_regularizer(reg_strength) if reg_strength > 0 else None

    tf.reset_default_graph()
    train_set, _, test_set = get_cifar10(data_dir)
    n_classes = train_set.labels.shape[1]
    m, width, height, depth = train_set.images.shape

    model = VGG(n_classes=n_classes, weight_regularizer=regularizer)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=(None, width, height, depth))
        y = tf.placeholder(tf.float32, shape=(None, n_classes))

    with tf.name_scope('test'):
        x_test = test_set.images
        y_test = test_set.labels


    forbid_propogation = tf.Variable(False, name="pred", dtype=bool)
    with tf.name_scope('inference'):
        tr_x, vgg_ops = load_pretrained_VGG16_pool5(x)
        tr_x = tf.cond(forbid_propogation, lambda: tf.stop_gradient(tr_x), lambda : tr_x)
        logits = model.inference(tr_x)
        tf.histogram_summary("logits", logits)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(model.loss(logits, y))
        tf.scalar_summary("loss", loss)

    with tf.name_scope('optimize'):
        train_stp = train_step(loss, optimizer_type="adam", learning_rate=learning_rate)

    with tf.name_scope('Accuracy'):
        accuracy = model.accuracy(logits, y)

    merge_op = tf.merge_all_summaries()
    # Initializing the variables
    init = tf.initialize_all_variables()


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    test_accuracies = []
    test_steps = []
    train_accuracies = []
    train_steps = []

    with tf.Session() as sess:
        run_var = get_run_var(log_dir)
        summary_writer = tf.train.SummaryWriter('%s/%s' % (log_dir, run_var))
        sess.run(init)
        sess.run(vgg_ops)

        print("starting training")

        for step in range(1, max_steps+1):
            forbid_propogation = tf.assign(forbid_propogation, tf.less_equal(tf.constant(step), tf.constant(refine_after_k)))
            x_batch, y_batch = train_set.next_batch(batch_size)
            _, loss_val, summary_str = sess.run([train_stp, loss, merge_op], feed_dict={x: x_batch, y: y_batch})
            if step % 10 == 0:
                print("Step:", '%04d' % step, "1 batch loss=", "{:.9f}".format(loss_val))
            if step % train_eval_freq == 0:
                train_acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
                train_accuracies.append(train_acc)
                train_steps.append(step)
                print("Step:", '%04d' % step, "train acc %f" % train_acc)
            if step % test_eval_freq == 0:
                test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
                test_accuracies.append(test_acc)
                test_steps.append(step)
                print("Step:", '%04d' % step, "test acc %f" % test_acc)
            if step % checkpoint_freq == 0:
                saver.save(sess, '%s/%s' % (checkpoint_dir, "model.ckpt"))
            summary_writer.add_summary(summary_str, step)

        summary_writer.flush()
        # Save the variables to disk.
        save_path = saver.save(sess, '%s/%s' % (checkpoint_dir, "model.ckpt"))
        print("Model saved in file: %s" % save_path)

        # plot
        save_accuracy_curve_in_a_file(train_accuracies, test_accuracies, train_steps, test_steps,
                                      file="plots/accuracy_ref_"+str(refine_after_k)+"_"+str(reg_strength)+".txt")
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--reg_strength', type = float, default = REGULARIZATION_STRENGTH_DEFAULT,
                      help='Regularization strength')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
