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
import matplotlib.gridspec as gridspec

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

# Class Definitions.
class NNConfig:
    """
    This class keeps all the configuration for running this network together at
    one spot so its easier to run it.
    """
    def __init__(self, EPOCHS, BATCH_SIZE, MAX_LABEL_SIZE, INPUT_LAYER_SHAPE,
                 LEARNING_RATE, SAVE_MODEL, NN_NAME, USE_AUGMENTED_FILE):
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
        self.USE_AUGMENTED_FILE = USE_AUGMENTED_FILE

        assert(len(INPUT_LAYER_SHAPE) == 3)

        self.NUM_CHANNELS_IN_IMAGE = INPUT_LAYER_SHAPE[2]


class TensorOps:
    """
    This class stores the tensor ops which are the end layers which we use for
    training.
    """
    def __init__(self, x, y, dropout_keep_prob, training_op, accuracy_op, loss_op, logits, saver):
        """
        x: Tensor for the input class
        y: Tensor for the output class
        training_op: Training operation Tensor
        accuracy_op: Tensor for the accuracy operation
        saver: Used for saving the eventual model
        """
        self.x = x
        self.y = y
        self.dropout_keep_prob = dropout_keep_prob
        self.training_op = training_op
        self.accuracy_op = accuracy_op
        self.loss_op = loss_op
        self.logits = logits
        self.saver = saver


class Image:
    @staticmethod
    def rotate_image(img, label):
        # Rotate the image by a random angle (-45 to 45 degrees)
        # Rotation has to be done within a very narrow range since it could
        # affect the meaning of the sign itself.
        # Choosing -10 to 10 degrees
        angle = np.random.choice(np.random.uniform(-10,10,100))
        dst = ndimage.rotate(img, angle)
        height, width = img.shape[:2]
        dst = cv2.resize(dst, (width, height))
        return dst

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
            flipped = cv2.flip(img, 1)
        elif label in can_flip_vertically:
            flipped = cv2.flip(img, 0)
        elif label in can_flip_both:
            flipped = cv2.flip(img, np.random.choice([-1, 0, 1]))

        return flipped

    @staticmethod
    def edge_detected(img, label):
        slice = np.uint8(img)
        canny = cv2.Canny(slice, 50, 150)
        backtorgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        return backtorgb

    @staticmethod
    def perform_random_op(img, label):
        ops = [Image.edge_detected, Image.flip_image,
               Image.translate_image, Image.rotate_image,
              ]

        random_op = ops[random.randint(0, len(ops) - 1)]
        print(str(random_op))
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

    @staticmethod
    def grayscale(image):
        # use lumnosity to convert to grayscale as done by GIMP software
        # refer https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
        image = image[:,:,0] * .21 + image[:,:,1] * .72 + image[:,:,2]* .07
        return image

    @staticmethod
    def normalize(data):
        return data / 255 * 0.8 + 0.1

    # iterate through the image set and convert them to grayscale images
    @staticmethod
    def preprocess(data):
        gray_images = []
        for image in data:
            gray = Image.grayscale(image)
            gray = np.reshape(gray,(32 , 32, 1))
            gray_images.append(gray)

        gray_images = np.array(gray_images)
        gray_images = Image.normalize(gray_images)

        return gray_images

class Data:
    """
    Encode the different data so its easier to pass them around
    """
    def __init__(self, X_train, y_train, X_validation, y_validation, X_test,
                 y_test, images_from_internet, filenames_from_internet):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test
        self.images_from_internet = images_from_internet
        self.filenames_from_internet = filenames_from_internet

        self.frame = pd.read_csv('signnames.csv')

    def preprocess(self):
        # Normalize the RGB values to 0.0 to 1.0
        self.X_train = Image.preprocess(self.X_train)
        self.X_test  = Image.preprocess(self.X_test)
        self.X_validation = Image.preprocess(self.X_validation)

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

    def visualize_training_data(self):
        _, height, width, channel = self.X_train.shape
        num_class = np.max(self.y_train) + 1

        training_data = np.concatenate((self.X_train, self.X_validation))
        training_labels = np.concatenate((self.y_train, self.y_validation))

        for c in range(0, num_class):
            print("Class %s" % c)
            indices = np.array(np.where(training_labels == c))[0]
            total_cols = 50
            total_rows = len(indices) / total_cols + 1

            results_image = 255. * np.ones(shape=(total_rows * height, total_cols * width, channel),
                                           dtype=np.float32)
            for n in range(len(indices)):
                sample_image = training_data[indices[n]]
                Image.insert_subimage(results_image, sample_image, (n / total_cols) * height, (n % total_cols) * width)

            filename = str(c) + ".png"
            cv2.imwrite('augmented/' + filename, cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
            print("Wrote image: %s" % filename)

    def _augment_data_for_class(self, label_id, augmented_size, training_labels, training_data):
        """
        Internal method which will augment the data size for the specified label.
        It will calculate the initial size and augment it to its size.
        """
        print("\nAugmenting class: %s" % label_id)

        # find all the indices for the label id
        indices = np.array(np.where(training_labels == label_id))[0]
        total_data_len = len(indices)

        if indices.shape == 0:
            return np.array([]), np.array([])

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

        new_training_data = np.array(new_training_data)
        new_training_label = np.array(new_training_label)

        return new_training_data, new_training_label

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
            if i in training_labels:
                args.append((i, augmentation_data_size, training_labels, training_data))

        results = pool.starmap(self._augment_data_for_class, args)
        pool.close()
        pool.join()

        features, labels = zip(*results)

        features = np.array(features)
        labels = np.array(labels)

        augmented_features = np.concatenate(features, axis=0)
        augmented_labels = np.concatenate(labels, axis=0)
        all_features = np.concatenate(np.array([training_data, augmented_features]), axis=0)
        all_labels = np.concatenate(np.array([training_labels, augmented_labels]), axis=0)

        all_features, all_labels = shuffle(all_features, all_labels)

        train = {}
        train['features'] = all_features
        train['labels'] = all_labels

        f = open('augmented/augmented.p', 'wb')
        pickle.dump(train, f, protocol=4)

def convolutional_layer(input, num_input_filters, num_output_filters, filter_shape,
                        strides, padding, mean, stddev, activation_func=None, name=""):
    conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape+(num_input_filters, num_output_filters), mean=mean, stddev=stddev), name+"_weights")
    conv_b = tf.Variable(tf.zeros(num_output_filters), name=name+"_bias")

    conv = tf.nn.conv2d(input, conv_W, strides, padding) + conv_b

    # Activation Layer
    if activation_func is not None:
        conv = activation_func(conv)

    print(name + ": " + str(conv.get_shape().as_list()))
    return conv, num_output_filters


def fully_connected_layer(input, input_size, output_size, mean, stddev,
                          activation_func, dropout_prob, name):
    fc_W = tf.Variable(tf.truncated_normal(shape=(input_size, output_size),
                       mean=mean, stddev=stddev), name=name + "_W")
    fc_b = tf.Variable(tf.zeros(output_size), name=name + "_b")
    fc   = tf.matmul(input, fc_W) + fc_b

    if activation_func is not None:
        fc = activation_func(fc, name=name + "_relu")

    fc = tf.nn.dropout(fc, dropout_prob)

    return fc, output_size


def maxpool2d(input, ksize, strides, padding, name=""):
    maxpool = tf.nn.max_pool(input, ksize, strides, padding, name=name)

    return maxpool

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def LeNet(x, dropout_keep_prob, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 28x28x6
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=6,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    # Maxpool Layer : Input 28x28x6                Output = 14x14x6
    maxpool1 = maxpool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="maxpool1")

    # Convolutional Layer : Input 14x14x6          Output = 10x10x16
    conv2, num_output_filters = convolutional_layer(maxpool1, num_input_filters=num_output_filters, num_output_filters=16,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='VALID',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_2")

    # Maxpool Layer : Input = 10x10x16             Output = 5x5x16
    maxpool2 = maxpool2d(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="maxpool2")

    # Fully Connected Layer
    fc0 = flatten(maxpool2)

    shape = fc0.get_shape().as_list()[1]

    # Layer 3: Fully Connected: Input = 400           Output = 120
    fc1, shape = fully_connected_layer(fc0, shape, 120, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc1")

    # Layer 4: Fully Connected: Input = 120           Output = 84
    fc2, shape = fully_connected_layer(fc1, shape, 84, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc2")

    # logits
    # MAKE SURE LOGITS HAS NO DROPOUT
    logits, _ = fully_connected_layer(fc2, shape, cfg.MAX_LABELS, mu, sigma,
                                      activation_func=None, dropout_prob=1.0,
                                      name="logits")

    # Create a Network param dict for visualization
    network_params = {
        "conv1": conv1,
        "maxpool1": maxpool1,
        "conv2": conv2,
        "maxpool2": maxpool2,
        "fc0": fc0,
        "fc1": fc1,
        "fc2": fc2,
        "logits": logits
    }

    cfg.NETWORK_PARAMS = network_params

    return logits

def simple_1conv_layer_nn(x, dropout_keep_prob, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 26x26x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=12,
                                                    filter_shape=(7, 7), strides=[1,1,1,1], padding='VALID',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    # Now use a fully connected Layer
    fc0 = flatten(conv1)

    shape = fc0.get_shape().as_list()[1]

    # Use 2 more layers
    # Fully Connected: Input = 192               Output = 96
    fc1, output_size = fully_connected_layer(fc0, shape, 96, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc1")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc1, output_size, cfg.MAX_LABELS, mu, sigma,
                                                activation_func=None, dropout_prob=1.0, name="logits")

    # Create a Network param dict for visualization
    network_params = {
        "conv1": conv1,
        "fc0": fc0,
        "fc1": fc1,
        "logits": logits
    }

    cfg.NETWORK_PARAMS = network_params

    return logits

def simple_2conv_layer_nn(x, dropout_keep_prob, cfg):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input 32x32x3         Output = 26x26x12
    conv1, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=12,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    conv1 = maxpool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    conv2, num_output_filters = convolutional_layer(conv1, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                                                    filter_shape=(7, 7), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    maxpool1 = maxpool2d(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="maxpool1")

    # Now use a fully connected Layer
    fc0 = flatten(maxpool1)

    shape = fc0.get_shape().as_list()[1]

    # Use 2 more layers
    # Fully Connected: Input = 192               Output = 96
    fc1, output_size = fully_connected_layer(fc0, shape, 96, mu, sigma,
                                             activation_func=tf.nn.relu, dropout_prob=dropout_keep_prob, name="fc1")

    # Fully Connected: Input = 96               Output = 43
    logits, output_size = fully_connected_layer(fc1, 96, 43, mu, sigma,
                                                activation_func=None, dropout_prob=1.0, name="logits")

    return logits

def DeepNetMergeLayers(x, dropout_keep_prob, cfg):
    mu = 0
    sigma = 0.1

    conv1, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=3,
                                                    filter_shape=(1, 1), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")


    # Group 1
    conv2, num_output_filters = convolutional_layer(conv1, num_input_filters=num_output_filters, num_output_filters=8,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_2")

    conv3, num_output_filters = convolutional_layer(conv2, num_input_filters=num_output_filters, num_output_filters=8,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_3")

    maxpool1 = maxpool2d(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool1")

    dropout1 = tf.nn.dropout(maxpool1, dropout_keep_prob, name="dropout1")

    # Group 2
    conv4, num_output_filters = convolutional_layer(dropout1, num_input_filters=num_output_filters, num_output_filters=16,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_4")

    conv5, num_output_filters = convolutional_layer(conv4, num_input_filters=num_output_filters, num_output_filters=16,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_5")

    maxpool2 = maxpool2d(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool2")

    dropout2 = tf.nn.dropout(maxpool2, dropout_keep_prob, name="dropout2")

    # Group 3
    conv6, num_output_filters = convolutional_layer(dropout2, num_input_filters=num_output_filters, num_output_filters=32,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_6")

    conv7, num_output_filters = convolutional_layer(conv6, num_input_filters=num_output_filters, num_output_filters=32,
                                                    filter_shape=(5, 5), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_6")

    maxpool3 = maxpool2d(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool3")

    dropout3 = tf.nn.dropout(maxpool3, dropout_keep_prob, name="dropout3")

    # Now Flatten all the layers together
    layer_flat_group1, num_fc_layers_group1 = flatten_layer(dropout1)
    layer_flat_group2, num_fc_layers_group2 = flatten_layer(dropout2)
    layer_flat_group3, num_fc_layers_group3 = flatten_layer(dropout3)
    layer_flat = tf.concat(values=[layer_flat_group1, layer_flat_group2, layer_flat_group3], axis=1)

    num_fc_layers = num_fc_layers_group1 + num_fc_layers_group2 + num_fc_layers_group3

    fc_size1 = 1024
    ## FC_size
    fc_size2 = 1024

    # Fully Connected: Input = 1024               Output = 1024
    fc1, output_size = fully_connected_layer(layer_flat, num_fc_layers, num_fc_layers, mu, sigma,
                                             activation_func=tf.nn.relu, dropout_prob=dropout_keep_prob, name="fc1")

    fc2, output_size = fully_connected_layer(fc1, num_fc_layers, num_fc_layers, mu, sigma,
                                             activation_func=tf.nn.relu, dropout_prob=dropout_keep_prob, name="fc2")

    # Fully Connected: Input = 96               Output = 43
    logits, _ = fully_connected_layer(fc1, num_fc_layers, cfg.MAX_LABELS, mu, sigma,
                                      activation_func=None, dropout_prob=1.0, name="logits")

    return logits

def DeepNet(x, dropout_keep_prob, cfg):
    # Hyper parameters
    mu = 0
    sigma = 0.1

    one_by_one, num_output_filters = convolutional_layer(x, num_input_filters=cfg.NUM_CHANNELS_IN_IMAGE, num_output_filters=3,
                                                        filter_shape=(1, 1), strides=[1,1,1,1], padding='SAME',
                                                        mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_1")

    # Convolutional Layer 1: Input 32x32x3         Output = 32x32x12
    conv1, num_output_filters = convolutional_layer(one_by_one, num_input_filters=3, num_output_filters=12,
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
    maxpool1 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool1")

    # Convolutional Layer 6: Input 16x16x384         Output = 8x8x384
    conv6, num_output_filters = convolutional_layer(maxpool1, num_input_filters=num_output_filters, num_output_filters=num_output_filters * 2,
                                                    filter_shape=(11, 11), strides=[1,1,1,1], padding='SAME',
                                                    mean=mu, stddev=sigma, activation_func=tf.nn.relu, name="conv2d_6")

    # MaxPool Layer: Input 8x8x384                 Output = 4x4x384
    maxpool2 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool2")

    # Fully Connected Layer
    fc0 = flatten(maxpool2)

    # Fully Connected: Input = 6144                Output = 3072
    fc1, output_size = fully_connected_layer(fc0, 6144, 3072, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc1")

    # Fully Connected: Input = 3072                Output = 1536
    fc2, output_size = fully_connected_layer(fc1, 3072, 1536, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc2")

    # Fully Connected: Input = 1536               Output = 768
    fc3, output_size = fully_connected_layer(fc2, 1536, 768, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc3")


    # Fully Connected: Input = 768               Output = 384
    fc4, output_size = fully_connected_layer(fc3, 768, 384, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc4")

    # Fully Connected: Input = 384               Output = 192
    fc5, output_size = fully_connected_layer(fc4, 384, 192, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc5")

    # Fully Connected: Input = 192               Output = 96
    fc6, output_size = fully_connected_layer(fc5, 192, 96, mu, sigma, tf.nn.relu, dropout_keep_prob, name="fc6")

    # Fully Connected: Input = 96               Output = 43
    # logits, output_size = fully_connected_layer(fc6, 96, 43, mu, sigma, tf.nn.relu, dropout_keep_prob, name="logits")
    logits, output_size = fully_connected_layer(fc6, 96, cfg.MAX_LABELS, mu, sigma,
                                                activation_func=None, dropout_prob=dropout_keep_prob, name="logits")
    return logits

def train(cfg):
    print_header("Training " + cfg.NN_NAME + " --> Use Augmented Data: " + str(cfg.USE_AUGMENTED_FILE))
    cfg.IS_TRAINING = True
    global NETWORKS

    x = tf.placeholder(tf.float32, (None,) + cfg.INPUT_LAYER_SHAPE, name='X')
    y = tf.placeholder(tf.int32, (None), name='Y')
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    one_hot_y = tf.one_hot(y, cfg.MAX_LABELS)

    logits = NETWORKS[cfg.NN_NAME](x, dropout_keep_prob, cfg)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=one_hot_y)

    vars   = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * 0.001

    loss_operation = tf.reduce_mean(cross_entropy) + lossL2

    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE)
    training_op = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(), training_op, accuracy_op

    tensor_ops = TensorOps(x, y, dropout_keep_prob, training_op, accuracy_op, loss_operation, logits, saver)
    return tensor_ops

def evaluate(sess, X_data, y_data, tensor_ops, cfg):
    cfg.IS_TRAINING = False
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, cfg.BATCH_SIZE):
        batch_x = X_data[offset: offset + cfg.BATCH_SIZE]
        batch_y = y_data[offset: offset + cfg.BATCH_SIZE]

        accuracy, loss = sess.run([tensor_ops.accuracy_op, tensor_ops.loss_op],
                            feed_dict={tensor_ops.x: batch_x, tensor_ops.y: batch_y,
                                       tensor_ops.dropout_keep_prob: 1.0})

        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples, loss

# Data Loading and processing part
from os import listdir
from os.path import isfile, join

def load_traffic_sign_data(training_file, testing_file, preprocess):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    # Split the data into the training and validation steps.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    internet_test_set_path = 'internet_test_set'
    files_from_internet = [join(internet_test_set_path, f) for f in listdir(internet_test_set_path) if isfile(join(internet_test_set_path, f))]

    imgs_from_internet = []
    for f in files_from_internet:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        imgs_from_internet.append(img)

    imgs_from_internet = np.array(imgs_from_internet)

    if preprocess:
        X_train = Image.preprocess(X_train)
        X_test  = Image.preprocess(X_test)
        X_validation = Image.preprocess(X_validation)
        imgs_from_internet = Image.preprocess(imgs_from_internet)

    data = Data(X_train, y_train, X_validation, y_validation, X_test, y_test, imgs_from_internet, files_from_internet)

    return data


# Networks
NETWORKS = {
    "simple_nn1": simple_1conv_layer_nn,
    "simple_nn2": simple_2conv_layer_nn,
    "lenet": LeNet,
    "deepnetmergelayers": DeepNetMergeLayers,
    "deepnet": DeepNet
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

def predict(sess, tensor_ops, images, data, cfg, top_k=5):
    print("Predicting from Random Images: Number of Images: %s" % images.shape[0])
    cfg.IS_TRAINING = False
    pred = tf.nn.softmax(tensor_ops.logits)
    predictions = sess.run(pred, feed_dict={tensor_ops.x: images, tensor_ops.dropout_keep_prob: 1.0})
    values, indices = tf.nn.top_k(predictions, top_k)
    values, indices = values.eval(session=sess), indices.eval(session=sess)
    print(values, indices)

    filenames = data.filenames_from_internet
    for i, img in enumerate(images):
        plt.figure(figsize = (top_k, 1.5))
        gs = gridspec.GridSpec(1, 2,width_ratios=[2,3])
        plt.subplot(gs[0])
        if img.shape[2] == 1:
            img = np.reshape(img, (img.shape[0], img.shape[1]))
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(gs[1])
        plt.barh(top_k + 1 - np.arange(top_k), values[i], align='center')
        for i_label in range(top_k):
            plt.text(values[i][i_label] + .008, top_k + 1-i_label-.25, data.get_signname(indices[i][i_label]) + " --> (" + str(indices[i][i_label]) + ")")
            plt.text(values[i][i_label] / 2.0 - 0.01, top_k + 1-i_label-.25, "{:2.1f}%".format(values[i][i_label] * 100.0))
        plt.axis('off');
        plt.text(0,6.95, filenames[i].split(".")[0].split('/')[1]);
        plt.show();

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

    parser.add_argument("--preprocess", help="Preprocess the input data"
                        ,action="store_true")

    parser.add_argument("--save_model", help="Save the Final Model"
                        ,action="store_true")

    parser.add_argument("--dropout", help="Dropout probability for layers",
                        type=float, default=1.0)

    args = parser.parse_args()

    networks = []
    if args.network == "all":
        networks = NETWORKS.keys()
    else:
        networks = [args.network]

    if args.use_augmented_file:
        data = load_traffic_sign_data('augmented/augmented.p', 'data/test.p', args.preprocess)
    else:
        data = load_traffic_sign_data('data/train.p', 'data/test.p', args.preprocess)

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

    if args.augmentation_factor > 0:
        print_header("Starting Data Augmentation....")
        data.augment_data(args.augmentation_factor)
        print_header("Data Augmentation Complete.... Will Exit")
        return

    # data.normalize_data()
    dataframes = []
    for network in networks:
        df = pd.DataFrame(columns=('network name', 'epochs', 'validation accuracy', 'loss'))

        # Define the EPOCHS & BATCH_SIZE
        cfg = NNConfig(EPOCHS=args.epochs,
                       BATCH_SIZE=args.batch_size,
                       MAX_LABEL_SIZE=max_classified_id,
                       INPUT_LAYER_SHAPE=data.X_train[0].shape,
                       LEARNING_RATE=args.learning_rate,
                       SAVE_MODEL=args.save_model,
                       NN_NAME=network,
                       USE_AUGMENTED_FILE=args.use_augmented_file)
        tensor_ops = train(cfg)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print("Training...\n")
            for i in range(cfg.EPOCHS):
                X_train, y_train = shuffle(data.X_train, data.y_train)
                for offset in range(0, len(X_train), cfg.BATCH_SIZE):
                    end = offset + cfg.BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    batch_res, batch_loss = sess.run([tensor_ops.training_op, tensor_ops.loss_op],
                             feed_dict={tensor_ops.x: batch_x, tensor_ops.y: batch_y,
                                        tensor_ops.dropout_keep_prob: args.dropout})

                validation_accuracy, validation_loss = evaluate(sess, data.X_validation, data.y_validation,
                                                                tensor_ops, cfg)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))
                df.loc[i] = [network, i+1, "{:2.1f}".format(validation_accuracy * 100.0), validation_loss]


            test_accuracy, test_loss = evaluate(data.X_test, data.y_test, tensor_ops, cfg)
            print("Test Accuracy = {:.3f}\n".format(test_accuracy))
            df['test accuracy'] = "{:.3f}".format(test_accuracy)
            dataframes.append(df)

            if cfg.SAVE_MODEL is True:
                saver.save(sess, "./" + network)
                print("Model Saved")

            predict(sess, tensor_ops, data.images_from_internet, data, cfg)

    df = pd.concat(dataframes)
    print(df)
    df.to_csv('final_data.csv')
    df = pd.DataFrame.from_csv('final_data.csv')
    visualize_data(df)

if __name__ == "__main__":
    main()
