import tensorflow as tf
import numpy as np

import json
import pickle

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.contrib.layers import flatten

class Maxpool:
    def __init__(self, size, strides):
        self.size = size
        self.strides = strides

    def get_tensor(self, inputs):
        return tf.layers.max_pooling2d(inputs, self.size, self.strides)

    def __str__(self):
        return "Maxpool: (%s) -> (%s)" % (self.size, self.strides)

    __repr__ = __str__

class CNN:
    def __init__(self, name, filters, kernel_size, strides, padding, activation_func):
        self.name = name
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation_func = activation_func
        self.maxpool = None
        self.next = None

    def _activation_func(self):
        if self.activation_func == "relu":
            return tf.nn.relu

        return None

    def get_tensor(self, inputs):
        tensor = tf.layers.conv2d(
            inputs=inputs,
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self._activation_func()
        )

        if self.maxpool is not None:
            tensor = self.maxpool.get_tensor(tensor)

        return tensor

    def __str__(self):
        return self.name + " (CNN)"

    __repr__ = __str__

class Dense:
    def __init__(self, name, units, dropout=None):
        self.name = name
        self.units = units
        self.dropout = dropout
        self.next = None

    def get_tensor(self, inputs):
        tensor = tf.layers.dense(inputs=inputs, units=self.units)
        return tensor

    def __str__(self):
        return self.name + " (Dense)"

    __repr__ = __str__


class Flatten:
    def __init__(self, name):
        self.name = name
        self.next = None

    def get_tensor(self, inputs):
        tensor = flatten(inputs)
        return tensor

    def __str__(self):
        return self.name + " (Flatten)"


class Architecture:
    def __init__(self, filename):
        fp = open(filename, 'r')
        s = fp.read()
        arch_str = json.loads(s)

        node_map = {}

        self.nodes = []

        # Create the nodes first.
        for key in arch_str:
            k = arch_str[key]
            node = None
            if k['type'] == 'cnn':
                node = CNN(key, k['filters'], k['kernel_size'], k['strides'], k['padding'], k['activation_func'])
                # Setup the Maxpool Node
                node.maxpool = Maxpool(k['maxpool']['size'], k['maxpool']['strides'])
            elif k['type'] == 'flatten':
                node = Flatten(key)
            elif k['type'] == 'dense':
                dropout = k.get('dropout')
                node = Dense(key, k['units'], dropout)

            assert(node is not None)
            node_map[key] = node
            self.nodes.append(node)

        # Create the connections from the node
        for key in arch_str:
            k = arch_str[key]
            src_node = node_map[key]
            if k['connects_to'] is not None:
                target_node = node_map[k['connects_to']]
                src_node.next = target_node

        self._topological_sort()

    def _get_entry_node(self):
        return self.nodes[0]

    def _topological_sort(self):
        # Do a topological sort to find out the first node we need
        pass

    def generate_graph(self, features):
        entry_node = self._get_entry_node()

        # Now at this point we can recursively go through and find the tensor
        input = features
        node = entry_node
        tensor = None
        self.logits = None
        while node is not None:
            tensor = node.get_tensor(input)
            input = tensor
            node = node.next
            self.logits = tensor

    def get_logits(self):
        return self.logits


class Data:
    def __init__(self):
        training_file = 'data/train.p'
        validation_file= 'data/valid.p'
        testing_file = 'data/test.p'

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']

    def render_data(self):
        image_with_label = zip(self.X_train, self.y_train)
        seen_labels = set()

        fig = plt.figure(figsize=(200, 200))
        total_unique_labels = len(set(self.y_train))
        unique_rows = total_unique_labels // 5 + 1

        grid = ImageGrid(fig, 151,  # similar to subplot(141)
                        nrows_ncols=(unique_rows, 5),
                        axes_pad=0.05,
                        label_mode="1",
                        )

        i = 0
        for i_l in image_with_label:
            img, label = i_l
            if label not in seen_labels:
                im = grid[i].imshow(img)
                seen_labels.add(label)
                i += 1

        plt.show()

def main():
    EPOCHS = 10
    BATCH_SIZE = 128

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))

    # Load the data first and split it into training and testing sets
    data = Data()

    # For One hot encoding lets make the depth the maximum value of y so that way we have the right
    # encoding for now. However, this is inefficient since this is a sparse matrix.
    # TODO: Re-normalize the labels so that the min value in the label becomes 1 and the rest of the
    # values are represented with the differences.
    one_hot_y = tf.one_hot(y, max(data.y_train))

    # Load the architecture file and see what we can figure out from it.
    arch = Architecture("lenet.json")
    arch.generate_graph(x)

    logits = arch.get_logits()
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_operation = optimizer.minimize(loss_operation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(data.X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            x_train, y_train = shuffle(data.X_train, data.y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})


if __name__ == "__main__":
    main()
