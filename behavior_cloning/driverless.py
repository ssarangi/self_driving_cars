import pandas as pd
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPool2D

import argparse

def simple_network():
    print("Training Simple Network")
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def simple_conv_network():
    print("Training Simple Convolutional Network")
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def read_training_data(filename):
    df = pd.read_csv(filename)
    return df

def load_data(df):
    all_samples = []
    measurements = []

    for index, row in df.iterrows():
        center_image = cv2.imread(row[0])
        left_image = cv2.imread(row[1])
        right_image = cv2.imread(row[2])

        all_samples.append(center_image)
        all_samples.append(left_image)
        all_samples.append(right_image)

        measurements.append(float(row[3]))
        measurements.append(float(row[3]))
        measurements.append(float(row[3]))

    return np.array(all_samples), np.array(measurements)

NETWORKS = {
    "simple": simple_network,
    "simple_conv": simple_conv_network,
}

def args_definition():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="Number of Epochs to train the network for"
                        ,type=int, default=20)

    parser.add_argument("--network", help="Specify which Neural Network to execute"
                        ,choices=list(NETWORKS.keys()) + ["all"], default="simple_network")

    args = parser.parse_args()
    return args

def main():
    global NETWORKS
    args = args_definition()
    df = read_training_data('data/track1/driving_log.csv')
    frames, steering_angles = load_data(df)
    model = NETWORKS[args.network]()
    model.fit(frames,
              steering_angles,
              validation_split=0.2,
              shuffle=True,
              epochs=args.epochs)

    model_name = args.network + '.h5'
    model.save(model_name)

if __name__ == "__main__":
    main()
