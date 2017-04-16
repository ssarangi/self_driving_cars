import pandas as pd
import numpy as np
import cv2
import sys
import os

from keras.models import Sequential, Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import (Flatten, Dense, Convolution2D, MaxPool2D,
 BatchNormalization, Dropout, Activation, Cropping2D, Lambda)
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf as ktf
from keras.models import load_model

from scipy.misc import imread
import scipy

import argparse

import matplotlib
import matplotlib.pyplot as plt

def read_image(filename):
    img = imread(filename).astype(np.float32)
    img = scipy.misc.imresize(img, 50)
    return img

def visualize_model_layer_output(filename, model, layer_name, output_folder):
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    model2 = Model(input=model.input, output=model.get_layer(layer_name).output)

    img = read_image(filename)
    img = np.expand_dims(img, axis=0)

    conv_features = model2.predict(img)
    print(filename)

    # plot features
    fig, _ = plt.subplots(figsize=(5, 5))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(conv_features[0,:,:,i], cmap='gray')

    fname = filename.split("/")[-1].split('.')[0]
    fig.savefig(output_folder + fname + '.png')
    plt.close(fig)

def args_definition():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model to Load"
                        ,type=str)

    parser.add_argument("--image", help="Image to be loaded"
                        ,type=str)

    parser.add_argument("--layer", help="Layer to be visualized"
                        ,type=str)

    args = parser.parse_args()
    return args

def main():
    args = args_definition()
    model = load_model(args.model)
    files = os.listdir('runs/track1_final_run/')
    for f in files:
        visualize_model_layer_output('runs/track1_final_run/' + f, model, args.layer, 'runs/track1_layers/')

if __name__ == "__main__":
    main()
