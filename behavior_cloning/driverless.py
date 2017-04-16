import pandas as pd
import numpy as np
import cv2
import sys
import os

from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import (Flatten, Dense, Convolution2D, MaxPool2D,
 BatchNormalization, Dropout, Activation, Cropping2D, Lambda)
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf as ktf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.misc import imread
import scipy

import matplotlib
import matplotlib.pyplot as plt

import argparse
import json

matplotlib.style.use('ggplot')

########################### Utilities #########################################
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
###

############################# VISUALIZATION ####################################
def show_data_distribution(df):
    binwidth = 0.025

    # histogram before image augmentation
    plt.hist(df.steering_angle, bins=np.arange(min(df.steering_angle), max(df.steering_angle) + binwidth, binwidth))
    plt.title('Number of images per steering angle')
    plt.xlabel('Steering Angle')
    plt.ylabel('# Frames')
    plt.show()

############################### NETWORK ########################################
def nvidia_end_to_end(shape, l2_regularization_scale):
    print("Training Nvidia End To End of input shape %s" % str(shape))
    height = shape[0]
    crop_factor = 0.2 # Top 40% to be removed
    crop_size = (int)(crop_factor * height)
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_size, 0), (0, 0)), input_shape=shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(BatchNormalization(axis=1, input_shape=shape))

    model.add(Convolution2D(16, (3, 3), padding='valid', strides=(2, 2), activation='elu',
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Convolution2D(24, (3, 3), padding='valid', strides=(1, 2), activation='elu',
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Convolution2D(36, (3, 3), padding='valid', activation='elu',
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Convolution2D(48, (2, 2), padding='valid', activation='elu',
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Convolution2D(48, (2, 2), padding='valid', activation='elu',
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Flatten())
    model.add(Dense(512,
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Dropout(.5))
    model.add(Activation('elu'))

    model.add(Dense(10,
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.add(Activation('elu'))

    model.add(Dense(1,
              kernel_regularizer=l2(l2_regularization_scale),
              bias_regularizer=l2(l2_regularization_scale)))

    model.summary()
    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)
    return model

################################# Dataset Manipulation Functions ##############################
def flip_image(img):
    fimg = np.fliplr(img)
    return fimg

def read_image(filename):
    img = imread(filename).astype(np.float32)
    img = scipy.misc.imresize(img, 50)
    return img

def change_brightness(img):
    change_pct = random.uniform(0.4, 1.2)

    # Change to HSV so as to change the brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:,:,2] * change_pct

    # Convert back to RGB
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

def read_csv(filename, cols):
    print("Reading Training file: %s" % filename)
    return pd.read_csv(filename, names=cols)

def drop_zero_value_steering_angle_rows(df, drop_to):
    """
    df: The dataframe to drop rows from
    col_name: The column to check from for steering_angle
    drop_to: How many rows to drop to
    """
    # print("Total rows: %s" % len(df))
    # indices = df[df[col_name] == 0.0].index
    # total_existing = indices.shape[0]
    # print("Total Zero Value rows: %s" % total_existing)
    # print("Dropping %s rows from df" % (total_existing - drop_to))
    # remove_indices = np.random.choice(indices, size=total_existing - drop_to)
    # new_df = df.drop(remove_indices)
    # indices = new_df[new_df[col_name] == 0.0].index
    # print("Remaining zero value %s" % len(indices))
    #
    # print("Total rows: %s" % len(new_df))
    # print("Dropped %s rows" % (total_existing - drop_to))
    # assert(len(df) - len(new_df) == (total_existing - drop_to))
    # return new_df
    df_with_zero = df[df.steering_angle == 0]
    df_without_zero = df[df.steering_angle != 0]
    df_with_zero = df_with_zero.sample(n=drop_to)
    new_df = pd.concat([df_with_zero, df_without_zero])
    return new_df

def align_steering_angles_data(df):
    """
    Given a dataframe drop the 0 value steering angles to bring it at par
    """
    new_df = drop_zero_value_steering_angle_rows(df, 600)
    return new_df

#############################  Data Reading Routines #################################
def read_training_data(track):
    cols = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'brake', 'speed']
    data_dirs = [entry.path for entry in os.scandir('data') if entry.is_dir()]

    dfs = []
    for ddir in data_dirs:
        # Ignore the recovery tracks since they will be loaded later
        if "recovery" not in ddir:
            if track in ddir:
                dfs.append(read_csv(ddir + '/driving_log.csv', cols))
            elif track == "both":
                dfs.append(read_csv(ddir + '/driving_log.csv', cols))

    df = pd.concat(dfs)
    return df

def read_sample_training(df):
    """
    df: Original DF from our training data which is to be augmented
    """
    cols = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'brake', 'speed']
    sample_df = read_csv('sample_training_data/driving_log.csv', cols)
    df = pd.concat([df, sample_df])
    return df

def preprocess(img):
    return img

def augment_image(img, technique):
    if technique == "flip":
        return flip_image(img)
    elif technique == "brightness":
        return change_brightness(img)

    assert("No Valid technique passed for image augmentation")

def load_data(df):
    all_samples = []
    measurements = []
    shape = None

    total_images = len(df)
    index = 0
    for i, row in df.iterrows():
        print_progress_bar(index, total_images)
        index += 1
        center_image = preprocess(read_image(row[0]))
        all_samples.append(center_image)
        measurements.append(float(row[3]))

        left_image = preprocess(read_image(row[1]))
        all_samples.append(left_image)
        measurements.append(float(row[3]) + (0.25))

        right_image = preprocess(read_image(row[2]))
        all_samples.append(right_image)
        measurements.append(float(row[3]) - (0.25))

        shape = center_image.shape

        # Add an image for the flipped version of the center image
        flipped_center_image = flip_image(center_image)
        all_samples.append(flipped_center_image)
        measurements.append(-float(row[3]))

    return np.array(all_samples), np.array(measurements), shape

# def setup_probabilistic_distribution(df):
#     binwidth = 0.025

#     num_bins = int((max(df.steering_angle) - min(df.steering_angle)) / binwidth)

#     # histogram before image augmentation
#     counts, bins = np.histogram(df['steering_angle'])

#     total = len(df.index)

def rearrange_and_augment_dataframe(df, shuffle_data):
    """
    Rearrange the dataframe to linearize the steering angle images and also add
    a column to indicate whether augmentation is required or not and what kind of
    augmentation is required.
    """
    center_df = pd.DataFrame()
    left_df = pd.DataFrame()
    right_df = pd.DataFrame()
    flipped_center = pd.DataFrame()

    center_df['image'] = df['center_image']
    flipped_center['image'] = df['center_image']

    left_df['image'] = df['left_image']
    right_df['image'] = df['right_image']

    center_df['steering_angle'] = df['steering_angle']
    left_df['steering_angle'] = df['steering_angle'] + 0.25
    right_df['steering_angle'] = df['steering_angle'] - 0.25
    flipped_center['steering_angle'] = -1.0 * df['steering_angle']

    # Set the dataframe columns for augmentation to false for some
    center_df['augmentation'] = False
    left_df['augmentation'] = False
    right_df['augmentation'] = False
    flipped_center['augmentation'] = True

    # Set the augmentation techniques we need
    center_df['techniques'] = ""
    left_df['techniques'] = ""
    right_df['techniques'] = ""
    flipped_center['techniques'] = "flip"

    # Change the brightness for images with different steering angles and add them
    brightness_df = center_df.loc[(center_df.steering_angle < -0.025) | (center_df.steering_angle > 0.025)]
    BRIGHTNESS_AUG_FACTOR = 20
    brightness_df = brightness_df.append([brightness_df]*BRIGHTNESS_AUG_FACTOR, ignore_index=True)
    brightness_df.steering_angle = brightness_df.steering_angle + (np.random.uniform(-1, 1)/30.0)

    new_df = pd.concat([center_df, left_df, right_df, flipped_center, brightness_df])

    if shuffle_data:
        shuffle(new_df)

    return new_df

def read_recovery_track_data():
    # Read the recovery track data for track 2
    cols = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'brake', 'speed']
    df = read_csv('data/track2_recovery/driving_log.csv', cols)
    recovery_df = rearrange_and_augment_dataframe(df, shuffle_data=True)
    return recovery_df

def save_experiment(name, network_used, epochs, model, hist):
    # Based on the experiment name, save the history and the model for future use
    experiments_folder = "experiments/"
    history_filename = experiments_folder + name + ".json"
    fp = open(history_filename, 'w')
    json.dump(hist.history, fp)
    print(hist.history)
    fp.close()

    model_filename = experiments_folder + name + "_" + str(epochs) + "_epochs_" + network_used + '.h5'
    model.save(model_filename)
    print("Wrote History file: %s" % history_filename)
    print("Wrote Model file: %s" % model_filename)

NETWORKS = {
    "nvidia": nvidia_end_to_end,
}

################################# GENERATORS ###################################
def new_generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images = []
            angles = []

            for i, batch_sample in batch_samples.iterrows():
                img = read_image(batch_sample.image)
                steering_angle = float(batch_sample.steering_angle)
                augment = batch_sample.augmentation
                techniques = batch_sample.techniques

                if augment:
                    # Techniques should be setup like this for multiple ones
                    # flip,brightness
                    techniques = techniques.split(",")

                    for technique in techniques:
                        img = augment_image(img, technique)

                images.append(img)
                angles.append(steering_angle)

            X = np.array(images)
            y = np.array(angles)
            yield X, y

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images = []
            angles = []

            for i, batch_sample in batch_samples.iterrows():
                center_image = read_image(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                left_image = read_image(batch_sample[1])
                left_angle = float(batch_sample[3] + 0.25)
                images.append(left_image)
                angles.append(left_angle)

                right_image = read_image(batch_sample[0])
                right_angle = float(batch_sample[3] - 0.25)
                images.append(right_image)
                angles.append(right_angle)

            X = np.array(images)
            y = np.array(angles)
            yield shuffle(X, y)

def training_generator(samples, batch_size=32):
    num_samples = len(samples)

    images = []
    angles = []

    # Drop all the rows and just keep 10
    # drop_indices = np.random.choice(samples.index, size=len(samples.index) - 100, replace=False)
    # samples = samples.drop(drop_indices)
    # First create the proper training data.
    print("Creating Initial Training Data...")
    for i, batch_sample in samples.iterrows():
        center_image = read_image(batch_sample[0])
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)

        left_image = read_image(batch_sample[1])
        left_angle = float(batch_sample[3] + 0.25)
        images.append(left_image)
        angles.append(left_angle)

        right_image = read_image(batch_sample[0])
        right_angle = float(batch_sample[3] - 0.25)
        images.append(right_image)
        angles.append(right_angle)

        # Also flip the center image and change the steering angle.
        flipped_center_image = flip_image(center_image)
        images.append(flipped_center_image)
        angles.append(-center_angle)

    images = np.array(images)
    angles = np.array(angles)

    print("Feeding to Keras Generator...")
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zca_whitening=False,
        channel_shift_range=0.2,
        zoom_range=0.2)

    # datagen.fit(images)

    while 1:
        X, y = shuffle(images, angles)

        for X_train, y_train in datagen.flow(X, y, batch_size=batch_size):
            yield shuffle(X_train, y_train)

################################# MAIN METHODS #################################
def args_definition():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="Number of Epochs to train the network for"
                        ,type=int, default=20)

    parser.add_argument("--network", help="Specify which Neural Network to execute"
                        ,choices=list(NETWORKS.keys()) + ["all"], default="simple_network")

    parser.add_argument("--track", help="Specify which track data to use",
                        choices=["track1", "track2", "both"], default="both")

    parser.add_argument("--use_sample_training", help="Use the sample training data",
                        action='store_true')

    parser.add_argument("--experiment", help="Give the run an experiment name", type=str)

    parser.add_argument("--show_data_distribution", help="Show the data distribution for the training data",
                        action='store_true')

    args = parser.parse_args()
    return args

def main():
    global NETWORKS
    args = args_definition()
    df = read_training_data(args.track)
    if args.use_sample_training:
        df = read_sample_training(df)

    frames, steering_angles, shape = load_data(df)
    model = NETWORKS[args.network](shape)
    hist = model.fit(frames,
                     steering_angles,
                     validation_split=0.2,
                     shuffle=True,
                     epochs=args.epochs)

    model_name = args.network + '.h5'
    model.save(model_name)

    if args.experiment != "":
        save_experiment(args.experiment, args.network, model, hist)

    from keras import backend as K
    K.clear_session()

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def main_generator():
    global NETWORKS
    args = args_definition()
    df = read_training_data(args.track)
    if args.use_sample_training:
        df = read_sample_training(df)

    df = rearrange_and_augment_dataframe(df, shuffle_data=True)
    if args.track == "track2" or args.track == "both":
        recovery_df = read_recovery_track_data()
        df = pd.concat([df, recovery_df])

    # df = align_steering_angles_data(df)
    if args.show_data_distribution:
        show_data_distribution(df)
        return

    BATCH_SIZE = 512
    train_samples, validation_samples = train_test_split(df, test_size=0.2)
    print("Total Training Samples: %s" % len(train_samples.index))
    print("Total Validation Samples: %s" % len(validation_samples.index))

    train_generator = new_generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = new_generator(validation_samples, batch_size=BATCH_SIZE)

    shape = (80, 160, 3)
    l2_regularization = 1e-7
    model = NETWORKS[args.network](shape, l2_regularization)

    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
        ModelCheckpoint('latest_run/' + args.experiment + "_" + args.network + "_{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', save_best_only=True, verbose=1),
    ]

    hist = model.fit_generator(train_generator,
                               steps_per_epoch=len(df) // BATCH_SIZE + 1,
                               validation_data=validation_generator,
                               epochs=args.epochs,
                               validation_steps=10,
                               callbacks=callbacks)

    model_name = args.network + '.h5'
    model.save(model_name)

    if args.experiment != "":
        save_experiment(args.experiment, args.network, args.epochs, model, hist)

    from keras import backend as K
    K.clear_session()

if __name__ == "__main__":
    # main()
    main_generator()
