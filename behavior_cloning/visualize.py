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
from keras.models import load_model

from scipy.misc import imread
import scipy

import argparse

def args_definition():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model to Load"
                        ,type=str)

    parser.add_argument("--image", help="Image to be loaded"
                        ,type=str)

    args = parser.parse_args()
    return args

def read_image(filename):
    img = imread(filename).astype(np.float32)
    img = scipy.misc.imresize(img, 50)
    return img

def visualize_model_layer_output(image, model, layer_name):
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    
    model2 = Model(input=model.input, output=model.get_layer(layer_name).output)

    img = read_image(filename)
    img = np.expand_dims(img, axis=0)

    conv_features = model2.predict(img)
    print("conv features shape: ", conv_features.shape)

    # plot features
    plt.subplots(figsize=(5, 5))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(conv_features[0,:,:,i], cmap='gray')
    plt.show()

def main():
    args = args_definition()
    model = load_model(args.model)
    visualize_model_layer_output(image, model, layer_name)

if __name__ == "__main__":
    main()
