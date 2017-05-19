import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
import time
import argparse
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg

from clogger import *

matplotlib.style.use('ggplot')

import glob
class Frames:
    def __init__(self, num_frames):
        self._initialized = False
        self._num_frames = num_frames
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
        if len(self._prev_bboxes) == self._num_frames:
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

frames = Frames(14)


def read_image(filename):
    logger.debug("Reading an image")
    img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = mpimg.imread(filename)
    return img

def overlay_image(img1, img2):
    img1[0:img2.shape[0], 0:img2.shape[1]] = img2[:, :]
    return img1

def color_hist(img, nbins=32, bins_range=(0, 256)):
    logger.debug("Performing Color histogram")
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)[0]
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)[0]
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)[0]

    hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist))

    return hist_features

def extract_channel(img, channel):
    return img[:, :, channel]

def change_colorspace(img, color_space='RGB'):
    logger.debug("Performing Spatial Binning")
    color_space_img = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            color_space_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            color_space_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            color_space_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            color_space_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            color_space_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        raise Exception("Invalid colorspace specified")

    # Return the features
    return color_space_img

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def extract_hog_features(img, pix_per_cell=8, cell_per_block=2, orient=8, feature_vec=True):
    logger.debug("Extract hog features")
    features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=feature_vec)

    return features, hog_image

def create_final_feature_vector(features):
    logger.debug("Create final feature vector")
    # Now normalize the features to make sure they don't dominate others
    X_scaler = StandardScaler().fit(features)
    scaled_X = X_scaler.transform(features)

    return scaled_X

def extract_feature_from_image(img, spatial_size=(16, 16),
                     hist_bins=16, hist_range=(0, 256), hog_feature=None, hog_feature_vec=True):
    color_space = "HLS"
    orient = 8
    pix_per_cell = 4
    cell_per_block = 2

    img = change_colorspace(img, color_space)
    spatial_features = bin_spatial(img, spatial_size)
    hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
    hog_images = []

    if hog_feature is None:
        hog_feature = []
        for i in range(img.shape[2]):
            hf, hog_image = extract_hog_features(extract_channel(img, channel=i), orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, feature_vec=hog_feature_vec)
            hog_images.append(hog_image)
            hog_feature.extend(hf)

    # Create the Final Feature Vector
    final_feature_vector = np.concatenate((spatial_features, hist_features, hog_feature))
    return final_feature_vector, hog_images

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, spatial_size=(64, 64),
                     hist_bins=16, hist_range=(0, 256)):
    logger.debug("Extracting Features")
    # Create a list to append feature vectors to
    features = []
    labels = []
    for img, label in imgs:
        img = cv2.resize(img, spatial_size)
        feature_vector, _ = extract_feature_from_image(img, spatial_size, hist_bins, hist_range, hog_feature_vec=True)
        features.append(feature_vector)
        labels.append(label)
    return np.array(features), np.array(labels)

def train_model(features, labels):
    logger.info("Training Model")
    features, labels = shuffle(features, labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, )
    clf = svm.SVC(verbose=True)
    decision_tree_clf = tree.DecisionTreeClassifier()
    t = time.time()
    model = clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

    t = time.time()
    dt_model = decision_tree_clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(decision_tree_clf.score(X_test, y_test), 4))

    # Confusion Matrix
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    joblib.dump(clf, 'model.pkl')
    joblib.dump(decision_tree_clf, "decision_tree.pkl")

def create_training_data():
    logger.info("Creating Training Data")
    vehicles = []
    for filename in glob.iglob('training/vehicles/**/*.png', recursive=True):
        img = read_image(filename)
        # Flip the images to augment
        flipped = cv2.flip(img, 1)
        vehicles.append((img, 1))
        vehicles.append((flipped, 1))

    nonvehicles = []
    for filename in glob.iglob('training/non-vehicles/**/*.png', recursive=True):
        img = read_image(filename)
        flipped = cv2.flip(img, 1)
        nonvehicles.append((img, 0))
        nonvehicles.append((flipped, 0))

    return vehicles, nonvehicles

def visualize_training_data(vehicles, nonvehicles):
    logger.info("Visualize Training Data")
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'vehicles', 'non-vehicles'
    sizes = [len(vehicles), len(nonvehicles)]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

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

def generate_sliding_windows(img, window_size):
    height, width = img.shape[0], img.shape[1]
    x_start = width / 2 - 100
    x_stop = width
    y_start = height - 100
    y_stop = height // 2 + 60

    current_x = x_start
    current_y = y_start
    overlap = np.array([0.9, 0.9])

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

model_clf = None
decision_tree_clf = None
counter = 0

def prediction_pipeline(img):
    global counter, model_clf, decision_tree_clf, frames

    sizes = [256, 128, 96, 64]
    window_list = []
    # for size in sizes:
    #     wl = generate_sliding_windows(img, np.array([size, size]))
    #     window_list += wl
    window_list = generate_sliding_windows(img, np.array([64, 64]))

    frames.init(img)
    found_cars = []
    spatial_size = (16, 16)

    # if counter > 0:
    #     exit()
    # # Generate the HOG Features
    # orient = 8
    # pix_per_cell = 4
    # cell_per_block = 2
    #
    # hog_feature = []
    # for i in range(img.shape[2]):
    #     hf, _ = extract_hog_features(extract_channel(img, channel=i), orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, feature_vec=False)
    #     hog_feature.append(hf)

    for idx, window in enumerate(window_list):
        # In numpy the x & Y directions are reversed
        cropped = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        cv2.imwrite("cropped/" + str(idx) + ".png", cropped)
        cropped = cv2.resize(cropped, spatial_size)
        # Extract the Hog features from the entire hog features
        # hog_feat1 = hog_feature[0][window[0][1]:window[1][1], window[0][0]:window[1][0]].ravel()
        # hog_feat2 = hog_feature[1][window[0][1]:window[1][1], window[0][0]:window[1][0]].ravel()
        # hog_feat3 = hog_feature[2][window[0][1]:window[1][1], window[0][0]:window[1][0]].ravel()
        #
        # window_hog_features = []
        # window_hog_features.extend(hog_feat1)
        # window_hog_features.extend(hog_feat2)
        # window_hog_features.extend(hog_feat3)

        features, _ = extract_feature_from_image(cropped, spatial_size)
        features = create_final_feature_vector(features)
        res = model_clf.predict(features)[0]
        res_decision_tree = decision_tree_clf.predict(features)[0]
        if res == 1 and res_decision_tree == 1:
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
    global model_clf, decision_tree_clf
    # Load the model
    clf = joblib.load('model.pkl')
    decision_tree_clf = joblib.load("decision_tree.pkl")
    model_clf = clf
    filename = 'project_video.mp4'
    clip = VideoFileClip(filename).subclip(25, 30)
    output_clip = clip.fl_image(prediction_pipeline)
    output_clip.write_videofile("output_" + filename, audio=False)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", help="Perform Training", type=str, choices=["train", "test"], default="train")

    args = parser.parse_args()
    return args

def main():
    args = argument_parser()

    if args.action == "train":
        vehicles, nonvehicles = create_training_data()
        # visualize_training_data(vehicles, nonvehicles)
        training_data = np.array(vehicles + nonvehicles)
        features, labels = extract_features(training_data, spatial_size=(16, 16))
        features = create_final_feature_vector(features)
        train_model(features, labels)
    elif args.action == "test":
        detection_on_video()

if __name__ == "__main__":
    main()