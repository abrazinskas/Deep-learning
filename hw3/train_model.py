from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import tensorflow as tf
import numpy as np
from cifar10_utils import get_cifar10
from cifar10_siamese_utils import get_cifar10 as get_cifar_10_siamese,create_dataset
from support import get_run_var, plot_confusion_matrix, plot_conv_weights, save_accuracy_curve,\
    save_loss_curve, save_loss_curve_in_a_file,save_accuracy_curve_in_a_file
from tensorflow.contrib.layers import regularizers
from convnet import ConvNet
from siamese import Siamese


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'adam'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = 'checkpoints'
REGULARIZATION_STRENGTH_DEFAULT = 0.


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
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
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
    reg_strength = params['reg_strength']

    regularizer = regularizers.l2_regularizer(reg_strength) if reg_strength > 0. else None

    tf.reset_default_graph()
    train_set, val_set, test_set = get_cifar10(data_dir)
    n_classes = train_set.labels.shape[1]
    m, width, height, depth = train_set.images.shape

    model = ConvNet(n_classes=n_classes, weight_regularizer=regularizer)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=(None, width, height, depth))
        y = tf.placeholder(tf.float32, shape=(None, n_classes))

    with tf.name_scope('test'):
        x_test = test_set.images
        y_test = test_set.labels

    with tf.name_scope('inference'):
        logits = model.inference(x)
        tf.histogram_summary("logits", logits)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(model.loss(logits, y))
        tf.scalar_summary("loss", loss)

    with tf.name_scope('optimize'):
        train_stp = train_step(loss)

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
        summary_writer = tf.train.SummaryWriter('%s/%s' % (log_dir, run_var), graph=tf.get_default_graph())
        sess.run(init)

        # compute initial accuracy
        acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print ("initial test accuracy is %f " % acc)

        for step in range(1, max_steps+1):
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
        # save_accuracy_curve(train_accuracies, test_accuracies, train_steps, test_steps, file="plots/accuracy.pdf")
        save_accuracy_curve_in_a_file(train_accuracies, test_accuracies, train_steps, test_steps,
                                      file="plots/cnn_acc_"+str(reg_strength)+".txt")

    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
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
    sufsara = bool(params['surfsara'])

    tf.reset_default_graph()


    train_set, _, test_set = get_cifar_10_siamese(data_dir, one_hot=False)
    val_set = create_dataset(test_set, num_tuples=100, batch_size=FLAGS.batch_size, fraction_same=0.2)

    del test_set

    m, width, height, depth = train_set.images.shape

    model = Siamese()

    with tf.name_scope('input'):
        x_channel1 = tf.placeholder(tf.float32, shape=(None, width, height, depth))
        x_channel2 = tf.placeholder(tf.float32, shape=(None, width, height, depth))
        y = tf.placeholder(tf.float32, shape=(None, ))

    with tf.name_scope('inference'):
        out_channel1 = model.inference(x_channel1, reuse=False)
        out_channel2 = model.inference(x_channel2, reuse=True)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(model.loss(out_channel1, out_channel2, y, margin=0.2), reduction_indices=0)
        # loss = model.loss(out_channel1, out_channel2, y, margin=0.2)

    with tf.name_scope('optimize'):
        train_stp = train_step(loss, learning_rate=learning_rate, optimizer_type="adam")


    merge_op = tf.merge_all_summaries()
    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # some arrays to save loss information
    train_loss = []
    val_loss = []
    train_steps = []
    val_steps = []
    with tf.Session() as sess:
        run_var = get_run_var(log_dir)
        summary_writer = tf.train.SummaryWriter('%s/%s' % (log_dir, run_var), graph=tf.get_default_graph())
        sess.run(init)

        for step in range(1, max_steps+1):
            x_batch1, x_batch2, y_batch = train_set.next_batch(batch_size)
            _, loss_tr, summary_str = sess.run([train_stp, loss, merge_op], feed_dict={x_channel1: x_batch1,
                                                                                        x_channel2: x_batch2, y: y_batch})
            summary_writer.add_summary(summary_str, step)

            if step % train_eval_freq == 0:
                train_loss.append(loss_tr)
                train_steps.append(step)
                print("Step:", '%04d' % step, "train loss=", "{:.9f}".format(loss_tr))

            if step % test_eval_freq == 0:
                v_loss = 0
                for val_batch in val_set:
                    v_loss += sess.run(loss, feed_dict={x_channel1: val_batch[0], x_channel2: val_batch[1], y: val_batch[2]})
                v_loss = v_loss/float(len(val_set)) # take the average
                val_loss.append(v_loss)
                val_steps.append(step)
                print("Step:", '%04d' % step, "val loss %f" % v_loss)

            if step % checkpoint_freq == 0:
                saver.save(sess, '%s/%s' % (checkpoint_dir, "model.ckpt"))

        summary_writer.flush()
        # Save the variables to disk.
        save_path = saver.save(sess, '%s/%s' % (checkpoint_dir, "model.ckpt"))
        print("Model saved in file: %s" % save_path)
    # save loss curve
    if sufsara:
        save_loss_curve_in_a_file(train_loss, val_loss, train_steps, val_steps, file="plots/task2/loss.txt")
    else:
        save_loss_curve(train_loss, val_loss, train_steps, val_steps, file="plots/task2/loss.png")
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    params = vars(FLAGS)
    model_type = params['train_model']
    data_dir = params['data_dir']

    namesp = "ConvNet" if model_type == "linear" else "Siamese"
    tensors_to_extract = ["inference/%s/fc1/Relu:0"%namesp,
                         "inference/%s/fc2/Relu:0"%namesp,
                         "inference/%s/flatten/Reshape:0"%namesp, # flatten layer after reshaping
                         ]

    if model_type == "siamese":
        tensors_to_extract.append("inference/Siamese/l2_normalize:0")

    save_folder = "features"
    _, _, test_set = get_cifar10(data_dir)
    n_classes = test_set.labels.shape[1]
    m, width, height, depth = test_set.images.shape

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=(None, width, height, depth))
        y = tf.placeholder(tf.float32, shape=(None, n_classes))

    model = ConvNet(n_classes=10) if model_type == "linear" else Siamese()

    with tf.name_scope('inference'):
        logits = model.inference(x)

    x_test = test_set.images
    y_test = test_set.labels

    if model_type == "linear":
        with tf.name_scope('Accuracy'):
            accuracy = model.accuracy(logits, y)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_dir+"/model.ckpt")
        graph = tf.get_default_graph()

        # just for test purposes
        if model_type == "linear":
            acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            print("acc is %f" % acc)

        for t in tensors_to_extract:
            tensor = graph.get_tensor_by_name(t)
            vals = sess.run(tensor, feed_dict={x: x_test})
            s = re.sub("/|:", "_", t)
            np.save(save_folder+"/"+s, vals)

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
    if FLAGS.is_train=="True":
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

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
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')
    parser.add_argument('--surfsara', type = str, default = 'False',
                      help='whether execution is performed on surfsara clusters')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
