# -*- coding: latin1 -*-
import numpy as np

import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten

import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform

import argparse

import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt

from scipy import ndimage

import multiprocessing

matplotlib.style.use('ggplot')

import sys
def print_header(txt):
    print("-" * 100)
    print(txt)
    print("-" * 100)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def convolutional_layer(input, num_input_filters, num_output_filters, filter_shape,
                        strides, padding, mean, stddev, activation_func=None, name=""):
    conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape+(num_input_filters, num_output_filters), mean=mean, stddev=stddev), name+"_weights")
    conv_b = tf.Variable(tf.zeros(num_output_filters), name=name+"_bias")

    conv = tf.nn.conv2d(input, conv_W, strides, padding) + conv_b

    fc = tf.contrib.layers.batch_norm(conv,
                                      center=True, scale=True,
                                      is_training=True)
    # Activation Layer
    if activation_func is not None:
        conv = activation_func(conv)

    print(name + ": " + str(conv.get_shape().as_list()))
    return conv, num_output_filters


def fully_connected_layer(input, input_size, output_size, mean, stddev,
                          activation_func, dropout_prob, use_batch_normalization, name):
    fc_W = tf.Variable(tf.truncated_normal(shape=(input_size, output_size),
                       mean=mean, stddev=stddev), name=name + "_W")
    fc_b = tf.Variable(tf.zeros(output_size), name=name + "_b")
    fc   = tf.matmul(input, fc_W) + fc_b

    if use_batch_normalization:
        fc = tf.contrib.layers.batch_norm(fc,
                                          center=True, scale=True,
                                          is_training=True)
    if activation_func is not None:
        fc = activation_func(fc, name=name + "_relu")

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
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    # Now use a fully connected Layer
    fc0 = flatten(conv1)

    shape = fc0.get_shape().as_list()[1]

    # Use 2 more layers
    # Fully Connected: Input = 192               Output = 96
    fc1, output_size = fully_connected_layer(fc0, shape, 96, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc1")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc1, output_size, 43, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="logits")

    return logits


def simple_2conv_layer_nn(x, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 26x26x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=12,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    conv1 = maxpool2d_and_dropout(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='VALID', dropout_prob=0.2)


    conv2, num_output_filters = convolutional_layer(conv1, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(7, 7), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, name="conv2d_1")

    conv2 = maxpool2d_and_dropout(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='VALID', dropout_prob=0.4)

    # Now use a fully connected Layer
    fc0 = flatten(conv1)

    shape = fc0.get_shape().as_list()[1]

    # Use 2 more layers
    # Fully Connected: Input = 192               Output = 96
    fc1, output_size = fully_connected_layer(fc0, shape, 96, mu, sigma, tf.nn.relu, 0.5, use_batch_normalization=cfg.IS_TRAINING, name="fc1")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc1, 96, 43, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="logits")

    return logits

def dense_net(x, cfg):
    mu = 0
    sigma = 0.1

    with tf.variable_scope('preprocess') as scope:
        preprocess_conv3x3 = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=8,
                            filter_shape=(3, 3), padding='SAME', mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv3x3")

        preprocess_conv1x1 = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=8,
                            filter_shape=(1, 1), padding='SAME', mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv1x1")



# AlexNet implementation
def AlexNet(x, cfg):
    # Hyper parameters
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 32x32x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=3, num_output_filters=12,
                               filter_shape=(3, 3), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    # Convolutional Layer 2: Input 32x32x12         Output = 28x28x24
    conv2, num_output_filters = convolutional_layer(conv1, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_2")

    # Convolutional Layer 3: Input 28x28x24         Output = 24x24x48
    conv3, num_output_filters = convolutional_layer(conv2, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_3")

    # Convolutional Layer 4: Input 24x24x48         Output = 16x16x96
    conv4, num_output_filters = convolutional_layer(conv3, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(9, 9), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_4")

    # Now lets add Convolutional Layers with downsampling
    conv5, num_output_filters = convolutional_layer(conv4, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(3, 3), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_5")

    # MaxPool Layer: Input 16x16x192                 Output = 16x16x384
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolutional Layer 6: Input 16x16x384         Output = 8x8x384
    conv6, num_output_filters = convolutional_layer(conv5, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                               filter_shape=(11, 11), strides=[1,1,1,1], padding='SAME',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_6")

    # MaxPool Layer: Input 8x8x384                 Output = 4x4x384
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully Connected Layer
    fc0 = flatten(conv6)

    print(fc0.get_shape())

    # Fully Connected: Input = 6144                Output = 3072
    fc1, output_size = fully_connected_layer(fc0, 6144, 3072, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc1")

    # Fully Connected: Input = 3072                Output = 1536
    fc2, output_size = fully_connected_layer(fc1, 3072, 1536, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc2")

    # Fully Connected: Input = 1536               Output = 768
    fc3, output_size = fully_connected_layer(fc2, 1536, 768, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc3")


    # Fully Connected: Input = 768               Output = 384
    fc4, output_size = fully_connected_layer(fc3, 768, 384, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc4")

    # Fully Connected: Input = 384               Output = 192
    fc5, output_size = fully_connected_layer(fc4, 384, 192, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc5")

    # Fully Connected: Input = 192               Output = 96
    fc6, output_size = fully_connected_layer(fc5, 192, 96, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc6")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc6, 96, 43, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="logits")

    return logits


def LeNet_new(x, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 28x28x6
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=6,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    # Maxpool Layer : Input 28x28x6                Output = 14x14x6
    maxpool1 = maxpool2d_and_dropout(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='VALID', dropout_prob=0.5)

    # Convolutional Layer : Input 14x14x6          Output = 10x10x16
    conv2, num_output_filters = convolutional_layer(maxpool1, num_input_filters=num_output_filters, num_output_filters=16,
                               filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                               mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_2")

    # Maxpool Layer : Input = 10x10x16             Output = 5x5x16
    maxpool2 = maxpool2d_and_dropout(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='VALID', dropout_prob=0.0)

    # Fully Connected Layer
    fc0 = flatten(maxpool2)

    shape = fc0.get_shape().as_list()[1]

    # Layer 3: Fully Connected: Input = 400           Output = 120
    fc1, shape = fully_connected_layer(fc0, shape, 120, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc1")

    # Layer 4: Fully Connected: Input = 120           Output = 84
    fc2, shape = fully_connected_layer(fc1, shape, 84, mu, sigma, tf.nn.relu, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="fc2")

    # logits
    logits, _ = fully_connected_layer(fc2, shape, cfg.MAX_LABELS, mu, sigma, None, 0.0, use_batch_normalization=cfg.IS_TRAINING, name="logits")

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
    print_header("Training " + cfg.NN_NAME)
    cfg.IS_TRAINING = True
    global NETWORKS

    x = tf.placeholder(tf.float32, (None,) + cfg.INPUT_LAYER_SHAPE, name='X')
    y = tf.placeholder(tf.int32, (None), name='Y')

    one_hot_y = tf.one_hot(y, cfg.MAX_LABELS)

    logits = NETWORKS[cfg.NN_NAME](x, cfg)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=one_hot_y)

    vars   = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

    loss_operation = tf.reduce_mean(cross_entropy) + lossL2

    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE)
    training_op = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(), training_op, accuracy_op

    tensor_ops = TensorOps(x, y, training_op, accuracy_op, loss_operation, saver)
    return tensor_ops


def evaluate(X_data, y_data, tensor_ops, cfg):
    cfg.IS_TRAINING = False
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, cfg.BATCH_SIZE):
        batch_x = X_data[offset: offset + cfg.BATCH_SIZE]
        batch_y = y_data[offset: offset + cfg.BATCH_SIZE]

        accuracy, loss = sess.run([tensor_ops.accuracy_op, tensor_ops.loss_op],
                            feed_dict={tensor_ops.x: batch_x, tensor_ops.y: batch_y})

        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples, loss


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
        self.IS_TRAINING = False

        assert(len(INPUT_LAYER_SHAPE) == 3)

        self.NUM_CHANNELS_IN_IMAGE = INPUT_LAYER_SHAPE[2]


class TensorOps:
    """
    This class stores the tensor ops which are the end layers which we use for
    training.
    """
    def __init__(self, x, y, training_op, accuracy_op, loss_op, saver):
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
        self.loss_op = loss_op
        self.saver = saver


class Image:
    @staticmethod
    def rotate_image(img, label):
        # Rotate the image by a random angle (-45 to 45 degrees)
        # Rotation has to be done within a very narrow range since it could
        # affect the meaning of the sign itself.
        # Choosing -10 to 10 degrees
        angle = np.random.choice(np.random.uniform(-10,10,100))
        return ndimage.rotate(img, angle)

    @staticmethod
    def translate_image(img, label):
        tx = np.random.choice(np.arange(10))
        ty = np.random.choice(np.arange(10))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        rows, cols, _ = img.shape
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    @staticmethod
    def flip_image(img, label):
        can_flip_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
        # Classes of signs that, when flipped vertically, should still be classified as the same class
        can_flip_vertically = np.array([1, 5, 12, 15, 17])
        # Classes of signs that, when flipped horizontally and then vertically,
        #  should still be classified as the same class
        can_flip_both = np.array([32, 40])

        flipped = None

        if label in can_flip_horizontally:
            flipped = cv2.flip(img, 0)
        elif label in can_flip_vertically:
            flipped = cv2.flip(img, 1)
        elif label in can_flip_both:
            flipped = cv2.flip(img, np.random.choice([-1, 0, 1]))

        return flipped

    @staticmethod
    def edge_detected(img, label):
        slice = np.uint8(img)
        return cv2.Canny(slice, 50, 150)

    @staticmethod
    def add_blur(img, label):
        return ndimage.gaussian_filter(img, sigma=random.randint(1, 4))

    @staticmethod
    def perform_random_op(img, label):
        ops = [Image.rotate_image, Image.edge_detected, Image.add_blur,
               Image.flip_image, Image.translate_image]

        random_op = ops[random.randint(0, len(ops) - 1)]
        new_img = random_op(img, label)
        while new_img is None:
            random_op = ops[random.randint(0, len(ops) - 1)]
            new_img = random_op(img, label)

        return new_img

    @staticmethod
    def insert_subimage(image, sub_image, y, x):
        h, w, c = sub_image.shape
        image[y:y+h, x:x+w, :]=sub_image
        return image


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

        self.frame = pd.read_csv('signnames.csv')

    def normalize_data(self):
        # Normalize the RGB values to 0.0 to 1.0
        self.X_train = self.X_train / 255.0
        self.X_validation = self.X_validation / 255.0
        self.X_test = self.X_test / 255.0

    def get_signname(self, label_id):
        return self.frame["SignName"][label_id]

    def display_statistics(self):
        """
        Figure out statistics on the data using Pandas.
        """
        _, height, width, channel = self.X_train.shape
        num_class = np.max(self.y_train) + 1

        training_data = np.concatenate((self.X_train, self.X_validation))
        training_labels = np.concatenate((self.y_train, self.y_validation))

        num_sample = 10
        results_image = 255.*np.ones(shape=(num_class*height, (num_sample + 2 + 22) * width, channel), dtype=np.float32)
        for c in range(num_class):
            indices = np.array(np.where(training_labels == c))[0]
            random_idx = np.random.choice(indices)
            label_image = training_data[random_idx]
            Image.insert_subimage(results_image, label_image, c * height, 0)

            #make mean
            idx = list(np.where(training_labels == c)[0])
            mean_image = np.average(training_data[idx], axis=0)
            Image.insert_subimage(results_image, mean_image, c * height, width)

            #make random sample
            for n in range(num_sample):
                sample_image = training_data[np.random.choice(idx)]
                Image.insert_subimage(results_image, sample_image, c*height, (2 + n) * width)

            #print summary
            count=len(idx)
            percentage = float(count)/float(len(training_data))
            cv2.putText(results_image, '%02d:%-6s'%(c, self.get_signname(c)), ((2+num_sample)*width, int((c+0.7)*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(results_image, '[%4d]'%(count), ((2+num_sample+14)*width, int((c+0.7)*height)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255), 1)
            cv2.rectangle(results_image,((2+num_sample+16)*width, c*height),((2+num_sample+16)*width + round(percentage * 3000), (c+1)*height),(0, 0, 255), -1)


        cv2.imwrite('augmented/data_summary.jpg',cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))


    def _augment_data_for_class(self, label_id, augmented_size, training_labels, training_data):
        """
        Internal method which will augment the data size for the specified label.
        It will calculate the initial size and augment it to its size.
        """
        print("\nAugmenting class: %s" % label_id)

        # find all the indices for the label id
        indices = np.array(np.where(training_labels == label_id))[0]
        total_data_len = len(indices)

        print("Label %s has %s images. Augmenting by %s images to %s images" % (label_id, total_data_len, (augmented_size - total_data_len), augmented_size))

        new_training_data = []
        new_training_label = []
        # Find a random ID from the indices and perform a random operation
        for i in range(0, (augmented_size - total_data_len)):
            print_progress_bar(i, (augmented_size - total_data_len), prefix='Progress:', suffix='Complete', bar_length=50)
            random_idx = np.random.choice(indices)
            img = training_data[random_idx]
            nimg = Image.perform_random_op(img=img, label=random_idx)

            # Add this to the training dataset
            new_training_data.append(nimg)
            new_training_label.append(label_id)


        return np.array(training_data), np.array(training_labels)

    def visualize_training_data(self):
        _, height, width, channel = self.X_train.shape
        num_class = np.max(self.y_train) + 1

        training_data = np.concatenate((self.X_train, self.X_validation))
        training_labels = np.concatenate((self.y_train, self.y_validation))

        for c in range(num_class):
            indices = np.array(np.where(training_labels == c))[0]
            total_cols = 50
            total_rows = len(indices) / total_cols + 1

            print(total_rows, total_cols, len(indices))

            results_image = 255. * np.ones(shape=(total_rows * height, total_cols * width, channel),
                                           dtype=np.float32)
            for n in range(len(indices)):
                sample_image = training_data[indices[n]]
                Image.insert_subimage(results_image, sample_image, (n / total_cols) * height, (n % total_cols) * width)

            filename = str(c) + "_" + self.get_signname(c) + ".png"
            cv2.imwrite('augmented/' + filename, cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
            print("Wrote image")


    def augment_data(self, augmentation_factor):
        """
        Augment the input data with more data so that we can make all the labels
        uniform
        """
        # Find the class label which has the highest images. We will decide the
        # augmentation size based on that multipled by the augmentation factor
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        training_labels = np.concatenate((self.y_train, self.y_validation))
        training_data   = np.concatenate((self.X_train, self.X_validation))

        bincounts = np.bincount(training_labels)
        label_counts = bincounts.shape[0]

        max_label_count = np.max(bincounts)
        augmentation_data_size = max_label_count * augmentation_factor

        print_header("Summary for Training Data for Augmentation")
        print("Max Label Count: %s" % max_label_count)
        print("Augmented Data Size: %s" % augmentation_data_size)

        args = []
        for i in range(0, label_counts):
            args.append((i, augmentation_data_size, training_labels, training_data))

        results = pool.starmap(self._augment_data_for_class, args)
        augmented_data = np.concatenate(results, axis=0)
        self.X_train = np.concatenate([self.X_train, augmented_data], axis=0)
        self.y_train = np.concatenate([self.y_train, augmented_data], axis=0)

        self.X_train = shuffle(self.X_train)
        self.y_train = shuffle(self.y_train)

        train = {}
        train['features'] = self.X_train
        train['labels'] = self.y_train

        f = open('augmented/augmented.p', 'wb')
        pickle.dump(train, f)

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

# Networks
NETWORKS = {
    "simple_nn1": simple_1conv_layer_nn,
    "simple_nn2": simple_2conv_layer_nn,
    "lenet": LeNet_new,
    # "alexnet": AlexNet
}

def visualize_data(df):
    """
    Takes in a Pandas Dataframe and then slices and dices it to create graphs
    """
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('validation accuracy')

    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')

    legend1 = ax1.legend(loc='upper center', shadow=True)
    legend2 = ax2.legend(loc='upper center', shadow=True)

    for i, group in df.groupby('network name'):
        group.plot(x='epochs', y='validation accuracy', ax=ax1, label=i, marker='o', linewidth=2)
        group.plot(x='epochs', y='loss', ax=ax2, label=i, marker='o', linewidth=2)

    plt.show()

def main():
    global NETWORKS

    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", help="Display Statistics for input data"
                        ,action="store_true")

    parser.add_argument("--print_training", help="Print the Training Data"
                        ,action="store_true")

    parser.add_argument("--epochs", help="Number of Epochs to train the network for"
                        ,type=int, default=20)

    parser.add_argument("--batch_size", help="Batch Size for training data"
                        ,type=int, default=128)

    parser.add_argument("--learning_rate", help="Learning Rate for Neural Network"
                        ,type=float, default=0.001)

    parser.add_argument("--network", help="Specify which Neural Network to execute"
                        ,choices=list(NETWORKS.keys()) + ["all"], default="simple_nn1")

    parser.add_argument("--augmentation_factor", help="Specify the data augmentation multiplier. Eg. amplify all input training data by 3 times"
                        ,type=int, default=0)

    parser.add_argument("--use_augmented_file", help="Use Augmented Training Data file"
                        ,action="store_true")

    args = parser.parse_args()

    networks = []
    if args.network == "all":
        networks = NETWORKS.keys()
    else:
        networks = [args.network]

    if args.use_augmented_file:
        data = load_traffic_sign_data('augmented/augmented.p', 'data/test.p')
    else:
        data = load_traffic_sign_data('data/train.p', 'data/test.p')

    # Find the Max Classified Id - For example, in MNIST data we have digits
    # from 0,..,9
    # Hence the max classified ID is 10
    # For the traffic sign database, the id's are encoded and max value is 42.
    # Hence the max classified ID is 43
    max_classified_id = np.max(data.y_train) + 1
    print("Max Classified id: %s" % (max_classified_id))

    if args.statistics is True:
        data.display_statistics()
        return

    if args.print_training is True:
        data.visualize_training_data()
        return

    data.normalize_data()

    if args.augmentation_factor > 0:
        print_header("Starting Data Augmentation....")
        data.augment_data(args.augmentation_factor)
        print_header("Data Augmentation Complete.... Will Exit")
        return

    dataframes = []
    for network in networks:
        df = pd.DataFrame(columns=('network name', 'epochs', 'validation accuracy', 'loss'))

        # Define the EPOCHS & BATCH_SIZE
        cfg = NNConfig(EPOCHS=args.epochs,
                       BATCH_SIZE=args.batch_size,
                       MAX_LABEL_SIZE=max_classified_id,
                       INPUT_LAYER_SHAPE=data.X_train[0].shape,
                       LEARNING_RATE=args.learning_rate,
                       SAVE_MODEL=False,
                       NN_NAME=network)
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

                validation_accuracy, validation_loss = evaluate(data.X_validation, data.y_validation,
                                                                tensor_ops, cfg)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))
                df.loc[i] = [network, i+1, "{:2.1f}".format(validation_accuracy * 100.0), validation_loss]


            test_accuracy, test_loss = evaluate(data.X_test, data.y_test, tensor_ops, cfg)
            print("Test Accuracy = {:.3f}\n".format(test_accuracy))
            df['test accuracy'] = "{:.3f}".format(test_accuracy)
            dataframes.append(df)

            if cfg.SAVE_MODEL is True:
                saver.save(sess, "./lenet")
                print("Model Saved")

    df = pd.concat(dataframes)
    print(df)
    df.to_csv('final_data.csv')
    df = pd.DataFrame.from_csv('final_data.csv')
    visualize_data(df)

if __name__ == "__main__":
    main()
