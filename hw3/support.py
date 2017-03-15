import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import math

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer. (a.k.a. filter depth)
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,   # Use 2x2 max-pooling.
                   name="conv1",  # the name of the layer
                   strides_conv=[1, 1, 1, 1],
                   strides_pool=[1, 2, 2, 1],
                   pool_ksize=[1, 2, 2, 1],
                   ac_fun=tf.nn.relu,  # activation function
                   weight_initializer=tf.random_normal_initializer(stddev=0.01)):

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = tf.get_variable(name=name+"_w", shape=shape, initializer=weight_initializer)

    # Create new biases, one for each filter.
    biases = tf.get_variable(name=name+"_b", shape=(num_filters, ), initializer=tf.constant_initializer(0.0))

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=strides_conv,
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    layer = ac_fun(layer)

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=pool_ksize,
                               strides=strides_pool,
                               padding='SAME')


    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer, num_features=None):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements() if num_features is None else num_features
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 name="fc1",
                 ac_fun=tf.nn.relu,  # activation function
                 weight_initializer=tf.random_normal_initializer(stddev=0.01)):

    # Create new weights and biases.
    weights = tf.get_variable(name=name+"_w", shape=[num_inputs, num_outputs], initializer=weight_initializer)
    biases = tf.get_variable(name=name+"_b", shape=(num_outputs, ), initializer=tf.constant_initializer(0.0))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    # print ("in layer: ")
    # print(input)
    # print(input.get_shape())
    # print(weights.get_shape())
    layer = tf.matmul(input, weights) + biases

    layer = ac_fun(layer)

    return layer, weights


def get_run_var(dir):
    subdirectories = get_immediate_subdirectories(dir)
    return len(subdirectories)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def plot_confusion_matrix(cls_pred, cls_true, num_classes=10):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.


    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_weights(weights, input_channel=0):

    model = TSNE(n_components=2, random_state=0)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = model.fit_transform(weights[:, :, input_channel, i])

            # Plot image.
            ax.imshow(img)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def save_accuracy_curve(train_acc, test_acc, train_iter_steps, test_iter_steps, file="accuracy.pdf"):
    plt.subplot()
    plt.plot(train_iter_steps, train_acc, '-o', label='train')
    plt.plot(test_iter_steps, test_acc, '-o', label='test')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.savefig(file)

def save_loss_curve(train_loss, val_loss, train_iter_steps, val_iter_steps, file="loss.pdf"):
    plt.subplot()
    plt.plot(train_iter_steps, train_loss, '-o', label='train')
    plt.plot(val_iter_steps, val_loss, '-o', label='val')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.savefig(file)


def save_loss_curve_in_a_file(train_loss, val_loss, train_iter_steps, val_iter_steps, file="loss.txt"):
    with open(file, 'w+') as f:
        f.write("train_iter_steps: "+str(train_iter_steps)+"\n")
        f.write("train_loss: "+str(train_loss)+"\n")
        f.write("val_iter_steps: "+str(val_iter_steps)+"\n")
        f.write("val_loss: "+str(val_loss)+"\n")


def save_accuracy_curve_in_a_file(train_acc, test_acc, train_iter_steps, test_iter_steps, file="acc.txt"):
    with open(file, 'w+') as f:
        f.write("train_iter_steps: "+str(train_iter_steps)+"\n")
        f.write("train_ac: "+str(train_acc)+"\n")
        f.write("test_iter_steps: "+str(test_iter_steps)+"\n")
        f.write("test_ac: "+str(test_acc)+"\n")