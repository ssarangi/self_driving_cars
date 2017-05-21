import argparse
import glob
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input
from keras.callbacks import TensorBoard
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import tqdm

from moviepy.editor import VideoFileClip

from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw, ImageFont

import multiprocessing

matplotlib.style.use('ggplot')

import logging

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = "[$BOLD%(name)-20s$RESET][%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    COLOR_FORMAT = formatter_message(FORMAT, True)
    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

####################################################### MAIN CODE ########################################################

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
        self._heatmap[self._heatmap <= threshold] = 0
        return self._heatmap

    def get_labels(self, bboxes, threshold):
        if len(self._prev_bboxes) > threshold:
            # Then remove the last bbox list from the previous frames
            self._prev_bboxes.pop(0)

        for pbboxes in self._prev_bboxes:
            self._add_heat(pbboxes)

        self._add_heat(bboxes)

        # Add the latest one
        self._prev_bboxes.append(bboxes)

        # Figure out the thresholded value
        # self._apply_threshold(threshold)
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

def overlay_text(image, text, pos=(0, 0), color=(255, 255, 255)):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./fonts/liberation-sans.ttf", 64)
    draw.text(pos, text, color, font=font)
    image = np.asarray(image)

    return image

def overlay_image(img1, img2):
    img1[0:img2.shape[0], 0:img2.shape[1]] = img2[:, :]
    return img1

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=2):
    # Make a copy of the image
    imcopy = np.copy(img)

    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    # Return the image copy with boxes drawn
    return imcopy

class LeNet:
    @staticmethod
    def build(width, height, depth, weightsPath=None):
        model = Sequential()
        # First set Conv Layers
        model.add(Conv2D(8, (3, 3), padding='valid', input_shape=(width, height, depth), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())

        # 2nd set Conv layers
        model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(width, height, depth), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Set of FC => Relu layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

class SimpleInception:
    @staticmethod
    def build(width, height, depth, weightsPath=None):
        input_img = Input(shape=(width, height, depth))
        model = Sequential()
        # First set Conv Layers
        tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
        tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
        tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
        tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)
        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)

        concat = concatenate([tower_1, tower_2, tower_3], axis=3)

        # Set of FC => Relu layers
        flatten = Flatten()(concat)
        dense1 = (Dense(256)(flatten))
        activation1 = Activation('relu')(dense1)
        dropout1 = Dropout(0.5)(activation1)

        # Softmax classifier
        dense2 = Dense(1)(dropout1)
        output = Activation('sigmoid')(dense2)

        model = Model(inputs=input_img, outputs=output)

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

def read_image(filename):
    logger.debug("Reading an image")
    img = mpimg.imread(filename)
    return img

def create_training_data():
    logger.info("Creating Training Data")
    vehicles = []
    for filename in tqdm.tqdm(glob.iglob('training/vehicles/**/*.png', recursive=True)):
        img = read_image(filename)
        vehicles.append(img)

    nonvehicles = []
    for filename in tqdm.tqdm(glob.iglob('training/non-vehicles/**/*.png', recursive=True)):
        img = read_image(filename)
        nonvehicles.append(img)

    return vehicles, nonvehicles

def train_model(vehicles, non_vehicles):
    generator = ImageDataGenerator( featurewise_center=True,
                                samplewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_std_normalization=False,
                                zca_whitening=False,
                                rotation_range=20.,
                                width_shift_range=0.4,
                                height_shift_range=0.4,
                                shear_range=0.2,
                                zoom_range=0.2,
                                channel_shift_range=0.1,
                                fill_mode='nearest',
                                horizontal_flip=True,
                                vertical_flip=False,
                                rescale=1.2,
                                preprocessing_function=None)

    logger.info("Training the Model")
    vehicles_labels = np.ones(len(vehicles))
    non_vehicles_labels = np.zeros(len(non_vehicles))
    labels = np.hstack((vehicles_labels, non_vehicles_labels))
    data = np.array(vehicles + non_vehicles)

    if len(vehicles[0].shape) == 3:
        width, height, depth = vehicles[0].shape[1], vehicles[0].shape[0], vehicles[0].shape[2]
    else:
        width, height, depth = vehicles[0].shape[1], vehicles[0].shape[0], 1

    # model = LeNet.build(width, height, depth)
    model = SimpleInception.build(width, height, depth)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, random_state=20)

    filepath = "inception.best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    generator.fit(trainData)
    hist = model.fit_generator(generator.flow(trainData, trainLabels, batch_size=16),
                      steps_per_epoch= int(len(trainData) / 16),
                      epochs=30,
                      verbose=1,
                      validation_data=(testData, testLabels),
                      callbacks=[TensorBoard(log_dir='logs'), checkpoint])

    print("[INFO] dumping weights to file...")
    model.save("inception.h5", overwrite=True)
    model.save_weights("inception_weights.hdf5", overwrite=True)

    fp = open("history_inception.json", 'w')
    json.dump(hist.history, fp)

def generate_sliding_windows_old(img, window_sizes):
    height, width = img.shape[0], img.shape[1]
    # x_start = width // 2 - 100
    x_start = 0
    x_stop = width
    y_start = height // 2 + 20
    y_stop = height - 70

    current_x = x_start
    current_y = y_start

    # Towards the bottom of the image use bigger bounding boxes
    window_list = []
    for (window_size, overlap) in window_sizes:
        while current_x < x_stop:
            end_x = current_x + window_size[0]
            while current_y < y_stop:
                end_y = current_y + window_size[1]
                window_list.append(((int(current_x), int(current_y)), (int(end_x), int(end_y))))
                current_y = end_y - window_size[1] * overlap[1]

            # At this point reset the x and update the y
            current_y = y_start
            current_x = end_x - (window_size[0] * overlap[0])

    return window_list

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def generate_sliding_windows(img):
    n_win_size = 3
    min_size = (30, 30)
    max_size = (120, 120)
    roi_upper = 380
    roi_lower = 650
    win_step_w = int((max_size[0] - min_size[0]) / n_win_size)
    win_step_h = int((max_size[1] - min_size[1]) / n_win_size)
    window_sizes = [(min_size[0] + i * win_step_w, min_size[1] + i * win_step_h) for i in range(n_win_size + 1)]

    all_windows = []
    for win_size in window_sizes:
        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[roi_upper, roi_lower],
                            xy_window=win_size, xy_overlap=(0.5, 0.5))
        all_windows += windows

    f = open('all_windows_mine.csv', 'w')
    for w in all_windows:
        f.write(str(w))
        f.write("\n")
    f.close()
    return all_windows

def crop_and_predict(idx, img, window, spatial_size):
    global model
    cropped = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    # cv2.imwrite("cropped/" + str(idx) + ".png", cropped)
    cropped = cv2.resize(cropped, spatial_size)
    cropped = np.array([cropped])
    return (window, cropped)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 5)
    # Return the image
    return img

counter = 0

window_sizes = [[(30, 30), (0.5, 0.5)],
                [(60, 60), (0.5, 0.5)],
                [(90, 90), (0.5, 0.5)],
                [(120, 120), (0.5, 0.5)]]

window_list = None
last_heat_map = None
use_n_frames = 3
def prediction_pipeline(img):
    global counter, model, frames, window_list, last_heat_map, use_n_frames
    # Normalize the image
    logger.debug("Scaling the image colors to 0-1")

    # window_list = generate_sliding_windows(img)
    if window_list is None:
        window_list = generate_sliding_windows(img)

    frames.init(img)
    spatial_size = (64, 64)

    prediction_images = []
    original_cropped_images = []
    for idx, window in enumerate(window_list):
        # In numpy the x & Y directions are reversed
        cropped = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        cropped_copy = np.copy(cropped)
        original_cropped_images.append(cropped_copy)
        cropped = cv2.resize(cropped, spatial_size)
        cropped = cropped / 255
        prediction_images.append(cropped)

    prediction_images = np.array(prediction_images)
    prediction = np.round(model.predict(prediction_images))

    found_cars = [window_list[i] for i in np.where(prediction == 1)[0]]
    found_car_idx = set([i for i in np.where(prediction == 1)[0]])

    for idx, window in enumerate(window_list):
        fname_prefix = "not_car"
        if idx in found_car_idx:
            fname_prefix = "car"

        mpimg.imsave("cropped/frame_" + str(counter) + "_" + str(idx) + "_" + fname_prefix + ".png", original_cropped_images[idx])

    # Now filter out the False positives
    # Define heatmap
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, found_cars)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    heatmap = heatmap.reshape(1, *heatmap.shape)

    if last_heat_map is None:
        last_heat_map = heatmap
        heatmap = last_heat_map[0]
    else:
        last_heat_map = last_heat_map[:use_n_frames, :]
        last_heat_map = np.concatenate([heatmap, last_heat_map])
        heatmap = last_heat_map.mean(axis=0)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    new_img = draw_labeled_bboxes(np.copy(img), labels)

    # found_cars, heatmap = frames.get_labels(found_cars, threshold=3)
    heatmap = cv2.resize(heatmap, (heatmap.shape[1] // 4, heatmap.shape[0] // 4))
    heatmap = np.dstack((heatmap, heatmap, heatmap)) * 255
    heatmap[:, :, 1] = 0
    heatmap[:, :, 2] = 0
    new_img = overlay_image(new_img, heatmap)

    # found_cars = window_list
    # new_img = draw_boxes(original, window_list)
    # new_img = draw_boxes(original, bboxes, color=(0, 1, 0), thick=2)

    mpimg.imsave('video_imgs/' + str(counter) + ".png", new_img)
    counter += 1
    return new_img

def detection_on_video():
    global model
    # Load the model
    # model = LeNet.build(64, 64, 3, 'weights.h5')
    # model = load_model('lenet.h5')
    model = load_model('inception.best.h5')

    filename = 'project_video.mp4'
    # clip = VideoFileClip(filename).subclip(21, 23)
    clip = VideoFileClip(filename)
    output_clip = clip.fl_image(prediction_pipeline)
    output_clip.write_videofile("output_" + filename, audio=False)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", help="Perform Training", type=str, choices=["train", "test", "test_img"], default="train")
    parser.add_argument("--file", help="File to perform action on", type=str)
    args = parser.parse_args()
    return args

def main():
    global model
    args = argument_parser()

    if args.action == "train":
        vehicles, nonvehicles = create_training_data()
        train_model(vehicles, nonvehicles)

    elif args.action == "test":
        detection_on_video()

    elif args.action == "test_img":
        model = load_model('inception.best.h5')
        img = read_image(args.file)
        new_img = prediction_pipeline(img) / 255
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(new_img)
        ax.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
