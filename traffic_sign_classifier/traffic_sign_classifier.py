import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import flatten

import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import argparse

# H = height, W = width, D = depth
#
# We have an input of shape 32x32x3 (HxWxD)
# 20 filters of shape 8x8x3 (HxWxD)
# A stride of 2 for both the height and width (S)
# Valid padding of size 1 (P)

# new_height = (input_height - filter_height + 2 * P)/S + 1
# new_width = (input_width - filter_width + 2 * P)/S + 1

def convolutional_layer(input, num_input_filters, num_output_filters, filter_shape,
                        strides, padding, mean, stddev, name):
    conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape+(num_input_filters, num_output_filters), mean=mean, stddev=stddev), name+"_weights")
    conv_b = tf.Variable(tf.zeros(num_output_filters), name=name+"_bias")

    conv = tf.nn.conv2d(input, conv_W, strides, padding) + conv_b

    # Activation Layer
    conv = tf.nn.relu(conv)

    print(name + ": " + str(conv.get_shape().as_list()))
    return conv, num_output_filters

def fully_connected_layer(input, input_size, output_size, mean, stddev,
                          activation_func, dropout_prob, name):
    fc_W = tf.Variable(tf.truncated_normal(shape=(input_size, output_size), mean=mean, stddev=stddev), name=name+"_W")
    fc_b = tf.Variable(tf.zeros(output_size), name=name+"_b")
    fc   = tf.matmul(input, fc_W) + fc_b

    if activation_func is not None:
        fc = activation_func(fc)

    if dropout_prob > 0.0:
        fc = tf.nn.dropout(fc, dropout_prob)

    return fc, output_size

def maxpool2d_and_dropout(input, ksize, strides, padding, dropout_prob):
    maxpool = tf.nn.max_pool(input, ksize, strides, padding)

    output = maxpool
    if dropout_prob > 0.0:
        dropout = tf.nn.dropout(maxpool, dropout_prob)

    return output

def simple_1conv_layer_nn(x, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 26x26x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=3, num_output_filters=12,
                               filter_shape=(7, 7), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, name="conv2d_1")

    # Now use a fully connected Layer
    fc0 = flatten(conv1)

    shape = fc0.get_shape().as_list()[1]

    # Use 2 more layers
    # Fully Connected: Input = 192               Output = 96
    fc1, output_size = fully_connected_layer(fc0, shape, 96, mu, sigma, tf.nn.relu, 0.0, "fc1")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc1, 96, 43, mu, sigma, tf.nn.relu, 0.0, "logits")

    return logits

def simple_2conv_layer_nn(x, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 26x26x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=3, num_output_filters=12,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, name="conv2d_1")

    conv1 = maxpool2d_and_dropout(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='VALID', dropout_prob=0.0)


    conv2, num_output_filters = convolutional_layer(conv1, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(7, 7), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, name="conv2d_1")

    conv2 = maxpool2d_and_dropout(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='VALID', dropout_prob=0.0)

    # Now use a fully connected Layer
    fc0 = flatten(conv1)

    shape = fc0.get_shape().as_list()[1]

    # Use 2 more layers
    # Fully Connected: Input = 192               Output = 96
    fc1, output_size = fully_connected_layer(fc0, shape, 96, mu, sigma, tf.nn.relu, 0.0, "fc1")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc1, 96, 43, mu, sigma, tf.nn.relu, 0.0, "logits")

    return logits


# AlexNet implementation
def AlexNet(x, cfg):
    # Hyper parameters
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 32x32x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=3, num_output_filters=12,
                               filter_shape=(3, 3), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, name="conv2d_1")

    # Convolutional Layer 2: Input 32x32x12         Output = 28x28x24
    conv2, num_output_filters = convolutional_layer(conv1, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, name="conv2d_2")

    # Convolutional Layer 3: Input 28x28x24         Output = 24x24x48
    conv3, num_output_filters = convolutional_layer(conv2, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, name="conv2d_3")

    # Convolutional Layer 4: Input 24x24x48         Output = 16x16x96
    conv4, num_output_filters = convolutional_layer(conv3, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(9, 9), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, name="conv2d_4")

    # Now lets add Convolutional Layers with downsampling
    conv5, num_output_filters = convolutional_layer(conv4, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(3, 3), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, name="conv2d_5")

    # MaxPool Layer: Input 16x16x192                 Output = 16x16x384
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolutional Layer 6: Input 16x16x384         Output = 8x8x384
    conv6, num_output_filters = convolutional_layer(conv5, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(11, 11), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, name="conv2d_6")

    # MaxPool Layer: Input 8x8x384                 Output = 4x4x384
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully Connected Layer
    fc0 = flatten(conv6)

    print(fc0.get_shape())

    # Fully Connected: Input = 6144                Output = 3072
    fc1, output_size = fully_connected_layer(fc0, 6144, 3072, mu, sigma, tf.nn.relu, 0.0, "fc1")

    # Fully Connected: Input = 3072                Output = 1536
    fc2, output_size = fully_connected_layer(fc1, 3072, 1536, mu, sigma, tf.nn.relu, 0.0, "fc2")

    # Fully Connected: Input = 1536               Output = 768
    fc3, output_size = fully_connected_layer(fc2, 1536, 768, mu, sigma, tf.nn.relu, 0.0, "fc3")


    # Fully Connected: Input = 768               Output = 384
    fc4, output_size = fully_connected_layer(fc3, 768, 384, mu, sigma, tf.nn.relu, 0.0, "fc4")

    # Fully Connected: Input = 384               Output = 192
    fc5, output_size = fully_connected_layer(fc4, 384, 192, mu, sigma, tf.nn.relu, 0.0, "fc5")

    # Fully Connected: Input = 192               Output = 96
    fc6, output_size = fully_connected_layer(fc5, 192, 96, mu, sigma, tf.nn.relu, 0.0, "fc6")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc6, 96, 43, mu, sigma, tf.nn.relu, 0.0, "logits")

    return logits


# LeNet implementation
def LeNet(x, cfg):
    # Hyper parameters
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input = 32x32x3        Output = 28x28x6
    conv1_W = tf.Variable(
        tf.truncated_normal(
            shape=(5, 5, cfg.NUM_CHANNELS_IN_IMAGE, 6),
            mean=mu, stddev=sigma), name="v1")

    conv1_b = tf.Variable(tf.zeros(6), name="v2")

    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation Layer
    conv1 = tf.nn.relu(conv1)

    print(conv1.get_shape())

    # Max Pooling : Input = 28x28x6 Output = 14x14x6
    conv1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer 2: Input = 14x14x6    Output: 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="v3")

    conv2_b = tf.Variable(tf.zeros(16), name="v4")

    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation Layer
    conv2 = tf.nn.relu(conv2)

    # Max Pooling : Input = 10x10x16       Output = 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='VALID')

    # Fully Connected Layer
    fc0 = flatten(conv2)

    # Layer 3 - Fully Connected: Input = 400     Output = 120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120),
                        mean=mu, stddev=sigma), name="v5")

    fc1_b = tf.Variable(tf.zeros(120), name="v6")
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation
    fc1 = tf.nn.relu(fc1)

    # Layer 4 : Fully Connected: Input = 120      Output = 84
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84),
                        mean=mu, stddev=sigma), name="v7")

    fc2_b = tf.Variable(tf.zeros(84), name="v8")
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation
    fc2 = tf.nn.relu(fc2)

    # Layer 5 - Fully Connected Input = 84         Output = Max Labels
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, cfg.MAX_LABELS),
                        mean=mu, stddev=sigma), name="v9")

    fc3_b = tf.Variable(tf.zeros(cfg.MAX_LABELS), name="v10")
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def train(cfg):
    x = tf.placeholder(tf.float32, (None,) + cfg.INPUT_LAYER_SHAPE, name='X')
    y = tf.placeholder(tf.int32, (None), name='Y')

    one_hot_y = tf.one_hot(y, cfg.MAX_LABELS)

    logits = cfg.NETWORKS[cfg.NN_NAME](x, cfg)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=one_hot_y)

    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE)
    training_op = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(), training_op, accuracy_op

    tensor_ops = TensorOps(x, y, training_op, accuracy_op, saver)
    return tensor_ops


def evaluate(X_data, y_data, tensor_ops, cfg):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, cfg.BATCH_SIZE):
        batch_x = X_data[offset: offset + cfg.BATCH_SIZE]
        batch_y = y_data[offset: offset + cfg.BATCH_SIZE]

        accuracy = sess.run(tensor_ops.accuracy_op,
                            feed_dict={tensor_ops.x: batch_x, tensor_ops.y: batch_y})

        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples


# Class Definitions.
class NNConfig:
    """
    This class keeps all the configuration for running this network together at
    one spot so its easier to run it.
    """
    def __init__(self, EPOCHS, BATCH_SIZE, MAX_LABEL_SIZE, INPUT_LAYER_SHAPE,
                 LEARNING_RATE, SAVE_MODEL, NN_NAME):
        """
        EPOCHS: How many times are we running this network
        BATCH_SIZE: How many inputs do we consider while running this data
        MAX_LABEL_SIZE: What is the maximum label size
                        (For example we have 10 classes for MNIST so its 10
                        For Traffic Database its 43)
        INPUT_LAYER_SHAPE: What is the shape of the input image.
                           How many channels does it have.
                           Eg. MNIST: 28x28x1
                           Traffic Sign: 32x32x3
        LEARNING_RATE: Learning rate for the network
        """
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LABELS = MAX_LABEL_SIZE
        self.INPUT_LAYER_SHAPE = INPUT_LAYER_SHAPE
        self.LEARNING_RATE = LEARNING_RATE
        self.SAVE_MODEL = SAVE_MODEL
        self.NN_NAME = NN_NAME

        assert(len(INPUT_LAYER_SHAPE) == 3)

        self.NUM_CHANNELS_IN_IMAGE = INPUT_LAYER_SHAPE[2]

        self.NETWORKS = {
            "simple_nn1": simple_1conv_layer_nn,
            "simple_nn2": simple_2conv_layer_nn,
            "lenet": LeNet,
            "alexnet": AlexNet
        }


class TensorOps:
    """
    This class stores the tensor ops which are the end layers which we use for
    training.
    """
    def __init__(self, x, y, training_op, accuracy_op, saver):
        """
        x: Tensor for the input class
        y: Tensor for the output class
        training_op: Training operation Tensor
        accuracy_op: Tensor for the accuracy operation
        saver: Used for saving the eventual model
        """
        self.x = x
        self.y = y
        self.training_op = training_op
        self.accuracy_op = accuracy_op
        self.saver = saver


class Data:
    """
    Encode the different data so its easier to pass them around
    """
    def __init__(self, X_train, y_train, X_validation, y_validation, X_test,
                 y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test


# Data Loading and processing part
def load_traffic_sign_data(training_file, testing_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    # Split the data into the training and validation steps.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    data = Data(X_train, y_train, X_validation, y_validation, X_test, y_test)
    return data


def main():
    data = load_traffic_sign_data('data/train.p', 'data/test.p')

    # parser = argparse.ArgumentParser(description='Process some integers.')
    #
    # parser.add_argument('epochs', metavar='Epochs', type=int, nargs='+',
    #                 help='an integer for the accumulator')
    #
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                 const=sum, default=max,
    #                 help='sum the integers (default: find the max)')
    #
    # args = parser.parse_args()
    # print(args.accumulate(args.integers))

    # Find the Max Classified Id - For example, in MNIST data we have digits
    # from 0,..,9
    # Hence the max classified ID is 10
    # For the traffic sign database, the id's are encoded and max value is 42.
    # Hence the max classified ID is 43
    max_classified_id = np.max(data.y_train) + 1
    print("Max Classified id: %s" % (max_classified_id))

    # Define the EPOCHS & BATCH_SIZE
    cfg = NNConfig(EPOCHS=200,
                   BATCH_SIZE=128,
                   MAX_LABEL_SIZE=max_classified_id,
                   INPUT_LAYER_SHAPE=data.X_train[0].shape,
                   LEARNING_RATE=0.001,
                   SAVE_MODEL=False,
                   NN_NAME="simple_nn2")

    tensor_ops = train(cfg)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...\n")
        for i in range(cfg.EPOCHS):
            X_train, y_train = shuffle(data.X_train, data.y_train)
            for offset in range(0, len(X_train), cfg.BATCH_SIZE):
                end = offset + cfg.BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(tensor_ops.training_op,
                         feed_dict={tensor_ops.x: batch_x, tensor_ops.y: batch_y})

            validation_accuracy = evaluate(data.X_validation, data.y_validation,
                                           tensor_ops, cfg)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))

        if cfg.SAVE_MODEL is True:
            saver.save(sess, "./lenet")
            print("Model Saved")

if __name__ == "__main__":
    main()