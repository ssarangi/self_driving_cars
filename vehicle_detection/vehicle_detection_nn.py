import argparse
import glob
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.models import load_model
import h5py

from moviepy.editor import VideoFileClip

from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split

class Frames:
    def __init__(self):
        self._initialized = False
        self._current_frame = 0
        self._prev_bboxes = []

    def init(self, img):
        self._heatmap = np.zeros_like(img)

    def _add_heat(self, bbox_list):
        for box in bbox_list:
            self._heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return self._heatmap

    def _apply_threshold(self, threshold):
        self._heatmap[self._heatmap < threshold] = 0
        return self._heatmap

    def get_heatmap(self):
        return self._heatmap

    def get_labels(self, bboxes, threshold):
        if len(self._prev_bboxes) == threshold:
            # Then remove the last bbox list from the previous frames
            self._prev_bboxes.pop(0)

        for pbboxes in self._prev_bboxes:
            self._add_heat(pbboxes)

        self._add_heat(bboxes)

        # Add the latest one
        self._prev_bboxes.append(bboxes)

        # Figure out the thresholded value
        self._apply_threshold(threshold)
        labels = label(self._heatmap)

        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)

        # Get a viewable heatmap
        heatmap = np.clip(self._heatmap, 0, 255)
        heatmap[heatmap[:, :, 0] > 0] += 100
        heatmap[:, :, 1] = 0
        heatmap[:, :, 2] = 0
        return bboxes, heatmap

frames = Frames()

def overlay_image(img1, img2):
    img1[0:img2.shape[0], 0:img2.shape[1]] = img2[:, :]
    return img1

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=2):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    color1 = (0, 0, 255)
    color2 = (255, 0, 0)
    color = color1
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        if color == color1:
            color = color2
        else:
            color = color1
    # Return the image copy with boxes drawn
    return imcopy

class LeNet:
    @staticmethod
    def build(width, height, depth, weightsPath=None):
        model = Sequential()
        # First set Conv Layers
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=(width, height, depth), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 2nd set Conv layers
        model.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Set of FC => Relu layers
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation('relu'))

        # Softmax classifier
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

from clogger import *

def read_image(filename):
    logger.debug("Reading an image")
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_training_data():
    logger.info("Creating Training Data")
    vehicles = []
    for filename in glob.iglob('training/vehicles/**/*.png', recursive=True):
        img = read_image(filename)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Flip the images to augment
        flipped = cv2.flip(gray, 1)

        gray = np.reshape(gray, (gray.shape[0], gray.shape[1], 1))
        flipped = np.reshape(flipped, (flipped.shape[0], flipped.shape[1], 1))

        vehicles.append(gray)
        vehicles.append(flipped)

    nonvehicles = []
    for filename in glob.iglob('training/non-vehicles/**/*.png', recursive=True):
        img = read_image(filename)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Flip the images to augment
        flipped = cv2.flip(gray, 1)

        gray = np.reshape(gray, (gray.shape[0], gray.shape[1], 1))
        flipped = np.reshape(flipped, (flipped.shape[0], flipped.shape[1], 1))

        nonvehicles.append(gray)
        nonvehicles.append(flipped)

    return vehicles, nonvehicles

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", help="Perform Training", type=str, choices=["train", "test"], default="train")

    args = parser.parse_args()
    return args

def train_model(vehicles, non_vehicles):
    logger.info("Training the Model")
    opt = SGD()
    vehicles_labels = np.ones(len(vehicles))
    non_vehicles_labels = np.zeros(len(non_vehicles))
    labels = np.hstack((vehicles_labels, non_vehicles_labels))
    data = np.array(vehicles + non_vehicles)

    if len(vehicles[0].shape) == 3:
        width, height, depth = vehicles[0].shape[1], vehicles[0].shape[0], vehicles[0].shape[2]
    else:
        width, height, depth = vehicles[0].shape[1], vehicles[0].shape[0], 1

    model = LeNet.build(width, height, depth)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    trainData, testData, trainLabels, testLabels = train_test_split(data / 255, labels, random_state=20)
    print(trainData.shape)
    print(trainLabels.shape)
    model.fit(trainData, trainLabels, batch_size=128, epochs=40, verbose=1,
              validation_data=(testData, testLabels),
              callbacks=[TensorBoard(log_dir='logs')])

    print("[INFO] dumping weights to file...")
    model.save("lenet.h5", overwrite=True)
    model.save_weights("lenet_weights.hdf5", overwrite=True)

def generate_sliding_windows(img, window_size):
    height, width = img.shape[0], img.shape[1]
    x_start = width / 2 - 100
    x_stop = width
    y_start = height - 100
    y_stop = height // 2 + 60

    current_x = x_start
    current_y = y_start
    overlap = np.array([0.5, 0.5])

    # Towards the bottom of the image use bigger bounding boxes
    # window_size = np.array([256, 128])
    window_list = []
    while current_y > y_stop:
        end_y = current_y - window_size[1]

        while current_x < x_stop:
            end_x = current_x + window_size[0]
            window_list.append(((int(current_x), int(end_y)), (int(end_x), int(current_y))))
            current_x = end_x - (window_size[0] * overlap[0])

        # At this point reset the x and update the y
        current_x = x_start
        current_y = current_y - window_size[1] * overlap[1]

        # Reduce the window size by 1/3
        # window_size = window_size * 0.9

    return window_list

counter = 0
def prediction_pipeline(img):
    global counter, model, frames

    sizes = [256, 128, 96, 64]
    window_list = generate_sliding_windows(img, np.array([64, 64]))

    frames.init(img)
    found_cars = []
    spatial_size = (64, 64)

    for idx, window in enumerate(window_list):
        # In numpy the x & Y directions are reversed
        cropped = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        cv2.imwrite("cropped/" + str(idx) + ".png", cropped)
        cropped = cv2.resize(cropped, spatial_size)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        cropped = np.reshape(cropped, (cropped.shape[0], cropped.shape[1], 1))

        new_shape = (1, cropped.shape[0], cropped.shape[1], cropped.shape[2])
        cropped = np.reshape(cropped, new_shape)
        res = model.predict(cropped, batch_size=1)
        if res == 1:
            found_cars.append(window)

    counter += 1

    # Now filter out the False positives
    found_cars, heatmap = frames.get_labels(found_cars, threshold=12)
    heatmap = cv2.resize(heatmap, (heatmap.shape[1] // 4, heatmap.shape[0] // 4))
    img = overlay_image(img, heatmap)
    # found_cars = window_list
    new_img = draw_boxes(img, found_cars)

    cv2.imwrite('video_imgs/' + str(counter) + ".png", new_img)
    counter += 1
    return new_img

def detection_on_video():
    global model
    # Load the model
    f = h5py.File('lenet.h5', mode='r')
    model = load_model('lenet.h5')

    filename = 'project_video.mp4'
    # clip = VideoFileClip(filename).subclip(25, 30)
    clip = VideoFileClip(filename)
    output_clip = clip.fl_image(prediction_pipeline)
    output_clip.write_videofile("output_" + filename, audio=False)

def main():
    args = argument_parser()

    if args.action == "train":
        vehicles, nonvehicles = create_training_data()
        train_model(vehicles, nonvehicles)

    elif args.action == "test":
        detection_on_video()

if __name__ == "__main__":
    main()