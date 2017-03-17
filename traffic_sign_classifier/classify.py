# Load pickled data
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


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

def LeNet(x, max_labels):
    # Hyper parameters
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1: Input = 32x32x3        Output = 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma), name="v1")
    conv1_b = tf.Variable(tf.zeros(6), name="v2")

    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation Layer
    conv1 = tf.nn.relu(conv1)

    # Max Pooling : Input = 28x28x6 Output = 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer 2: Input = 14x14x6    Output: 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="v3")
    conv2_b = tf.Variable(tf.zeros(16), name="v4")

    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation Layer
    conv2 = tf.nn.relu(conv2)

    # Max Pooling : Input = 10x10x16       Output = 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Fully Connected Layer
    fc0 = flatten(conv2)

    # Layer 3 - Fully Connected: Input = 400     Output = 120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name="v5")
    fc1_b = tf.Variable(tf.zeros(120), name="v6")
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation
    fc1 = tf.nn.relu(fc1)

    # Layer 4 : Fully Connected: Input = 120      Output = 84
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name="v7")
    fc2_b = tf.Variable(tf.zeros(84), name="v8")
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation
    fc2 = tf.nn.relu(fc2)

    # Layer 5 - Fully Connected Input = 84         Output = 10
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, max_labels), mean=mu, stddev=sigma), name="v9")
    fc3_b = tf.Variable(tf.zeros(max_labels), name="v10")
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

def train(max_classified_id):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="X")
    y = tf.placeholder(tf.int32, (None), name="Y")

    one_hot_y = tf.one_hot(y, max_classified_id)

    rate = 0.001

    logits = LeNet(x, max_classified_id)

    cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation     = tf.reduce_mean(cross_entropy)
    optimizer          = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(), training_operation, accuracy_operation
    return saver, training_operation, accuracy_operation, x, y

def evaluate(x, y, X_data, y_data, accuracy_operation, BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def main():
    data = Data()
    EPOCHS = 10
    BATCH_SIZE = 128

    max_classified_id = np.max(data.y_train)
    saver, training_operation, accuracy_operation, x, y = train(max_classified_id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(data.X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(data.X_train, data.y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(x, y, data.X_valid, data.y_valid, accuracy_operation, BATCH_SIZE)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, './lenet')
        print("Model saved")


if __name__ == "__main__":
    main()
