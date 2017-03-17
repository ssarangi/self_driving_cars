# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

import math


class FrameData:
    def __init__(self):
        self.left_lane_points = []
        self.right_lane_points = []
        self.left_lane_m = 0
        self.left_lane_c = 0
        self.right_lane_m = 0
        self.right_lane_c = 0

class Canny:
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

class Hough:
    def __init__(self, rho, theta, threshold, min_line_length, max_line_gap):
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

class Data:
    def __init__(self, canny_low_threshold, canny_high_threshold, rho, theta, threshold, min_line_length, max_line_gap):
        self.roi_vertices = None
        self.prev_frames_data = FrameData()
        self.canny = Canny(canny_low_threshold, canny_high_threshold)
        self.hough = Hough(rho, theta, threshold, min_line_length, max_line_gap)


class Config:
    def __init__(self,
                 max_frames_to_avg,
                 display_intermediate_imgs,
                 single_img_being_processed,
                 use_dominant_line_approach):
        self.MOVING_AVG_NUM_FRAME_DATA = max_frames_to_avg
        self.DISPLAY_INTERMEDIATE_IMGS = display_intermediate_imgs
        self.SINGLE_IMG_BEING_PROCESSED = single_img_being_processed
        self.USE_DOMINANT_LINE_APPROACH = use_dominant_line_approach


class Global:
    def __init__(self,
                 max_frames_to_avg=20,
                 display_intermediate_imgs=False,
                 single_img_being_processed=True,
                 use_dominant_line_approach=False,
                 canny_low_threshold=50,
                 canny_high_threshold=150,
                 rho=1,
                 theta=31 * np.pi/180,
                 threshold=10,
                 min_line_length=5,
                 max_line_gap=10):
        self.config = Config(max_frames_to_avg, display_intermediate_imgs, single_img_being_processed,
                             use_dominant_line_approach)
        self.data = Data(canny_low_threshold, canny_high_threshold, rho, theta, threshold, min_line_length, max_line_gap)


def find_line_equation(x0, y0, x1, y1):
    '''
    Find the line equation when the input is 2 points. (x0, y0) and (x1, y1)
    '''

    if x1 - x0 == 0:
        m = math.inf
    else:
        m = (y1 - y0) / (x1 - x0)

    c = y1 - m * x1

    return m, c


def find_line_length(x0, y0, x1, y1):
    length = math.hypot(x1 - x0, y1 - y0)
    return length


def divide_lines_into_2_clusters(lines, center_line_m, center_line_c, center_x=None):
    '''
    Hough transform returns all the lines from the line detection. Based on this center_x
    cluster the points into 2 sides. Left & Right
    '''
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if not math.isinf(center_line_m):
                center_x1 = (y1 - center_line_c) / (center_line_m)
                center_x2 = (y2 - center_line_c) / (center_line_m)
            else:
                center_x1 = center_x
                center_x2 = center_x

            m, _ = find_line_equation(x1, y1, x2, y2)

            # Ignore the horizontal lines
            if -0.5 < m < 0.5:
                continue

            if x1 <= center_x1 and x2 <= center_x2:
                left_lines.append(line)
            elif x1 > center_x1 and x2 > center_x2:
                right_lines.append(line)

    return left_lines, right_lines


def fit_single_line_to_set_of_points(points, dir, global_vals):
    '''
    Use Least squares method to fit a single line to a set of points
    '''
    # Divide the data points into 2 halves to see if we have outliers in the data or not.
    # Then figure out which one is closer to the slopes of the previous frame
    if dir == "left" and global_vals.config.SINGLE_IMG_BEING_PROCESSED is False:
        for p in global_vals.data.prev_frames_data.left_lane_points:
            points = points + p
    elif dir == "right" and global_vals.config.SINGLE_IMG_BEING_PROCESSED is False:
        for p in global_vals.data.prev_frames_data.right_lane_points:
            points = points + p

    v = np.asarray(points)
    try:
        x_coords = v[:, 0]
    except:
        x_coords = np.array([])

    try:
        y_coords = v[:, 1]
    except:
        y_coords = np.array([])

    # Use least squares fitting to a bunch of points
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords)[0]

    return m, c


def find_dominant_line(lines):
    longest_line_length = 0
    m = 0
    c = 0
    x = 0

    for line in lines:
        for x0, y0, x1, y1 in line:
            line_length = find_line_length(x0, y0, x1, y1)
            if line_length > longest_line_length:
                m_tmp, c_tmp = find_line_equation(x0, y0, x1, y1)
                if math.isinf(m_tmp):
                    x = x0

                # Ignore the line if it seems parallel to the x axis
                if not -0.5 < m_tmp < 0.5:
                    longest_line_length = line_length
                    m, c = m_tmp, c_tmp

    return m, c, x


def draw_hough_lines(lines, line_image, global_vals):
    '''
    Take a bunch of hough lines.
    1) Divide the hough lines into 2 parts based on center to left & right
    2) fit a single line for both left lane and right lane
    3) Draw the line
    '''
    # Numpy converts this roi_vertices into a 3d array for some reason
    roi_vertices = global_vals.data.roi_vertices[0]

    upper_y = roi_vertices[0][1]
    lower_y = roi_vertices[1][1] + 50

    lower_min_x = roi_vertices[0][0]
    lower_max_x = roi_vertices[3][0]

    lower_x = int((lower_min_x + lower_max_x) / 2)

    upper_min_x = roi_vertices[1][0]
    upper_max_x = roi_vertices[2][0]

    upper_x = int((upper_min_x + upper_max_x) / 2)

    center_line = [(lower_x, lower_y), (upper_x, upper_y)]

    # Find the center line equation
    x0, y0 = center_line[0]
    x1, y1 = center_line[1]
    center_line_m, center_line_c = find_line_equation(x0, y0, x1, y1)

    # If we have a line parallel to the Y-axis then just send the center x value
    center_line_x = None
    if center_line_m == math.inf:
        center_line_x = ((upper_min_x + upper_max_x) / 2)

    # Take the lines and divide them into 2 clusters.
    left_lines, right_lines = divide_lines_into_2_clusters(lines, center_line_m, center_line_c, center_line_x)

    if global_vals.config.USE_DOMINANT_LINE_APPROACH:
        m1, c1, x1 = find_dominant_line(left_lines)
    else:
        points = []
        for line in left_lines:
            for x1, y1, x2, y2 in line:
                points.append((x1, y1))
                points.append((x2, y2))

        m1, c1 = fit_single_line_to_set_of_points(points, "left", global_vals)

        if global_vals.config.SINGLE_IMG_BEING_PROCESSED is False:
            if len(global_vals.data.prev_frames_data.left_lane_points) == global_vals.config.MOVING_AVG_NUM_FRAME_DATA:
                global_vals.data.prev_frames_data.left_lane_points.remove(
                    global_vals.data.prev_frames_data.left_lane_points[0])

            global_vals.data.prev_frames_data.left_lane_points.append(points.copy())

    if global_vals.config.USE_DOMINANT_LINE_APPROACH:
        m2, c2, x2 = find_dominant_line(right_lines)
    else:
        points = []
        for line in right_lines:
            for x1, y1, x2, y2 in line:
                points.append((x1, y1))
                points.append((x2, y2))

        m2, c2 = fit_single_line_to_set_of_points(points, "right", global_vals)

        if global_vals.config.SINGLE_IMG_BEING_PROCESSED is False:
            if len(global_vals.data.prev_frames_data.right_lane_points) == global_vals.config.MOVING_AVG_NUM_FRAME_DATA:
                global_vals.data.prev_frames_data.right_lane_points.remove(
                    global_vals.data.prev_frames_data.right_lane_points[0])

            global_vals.data.prev_frames_data.right_lane_points.append(points.copy())

    if math.isinf(m1):
        left_x1 = x1
        left_x2 = x1
    else:
        if m1 == 0:
            m1 = global_vals.data.prev_frames_data.left_lane_m
            c1 = global_vals.data.prev_frames_data.left_lane_c

        left_x1 = (int)((lower_y - c1) / m1)
        left_x2 = (int)((upper_y - c1) / m1)

        global_vals.data.prev_frames_data.left_lane_m = m1
        global_vals.data.prev_frames_data.left_lane_c = c1

    if math.isinf(m2):
        right_x1 = x1
        right_x2 = x1
    else:
        if m2 == 0:
            m2 = global_vals.data.prev_frames_data.right_lane_m
            c2 = global_vals.data.prev_frames_data.right_lane_c

        right_x1 = (int)((lower_y - c2) / m2)
        right_x2 = (int)((upper_y - c2) / m2)

        global_vals.data.prev_frames_data.right_lane_m = m2
        global_vals.data.prev_frames_data.right_lane_c = c2

    # Draw 2 lines
    cv2.line(line_image, (left_x1, lower_y), (left_x2, upper_y), (255, 0, 0), 5)
    cv2.line(line_image, (right_x1, lower_y), (right_x2, upper_y), (255, 0, 0), 5)
    # for line in lines:
    #     for x0, y0, x1, y1 in line:
    #         cv2.line(line_image, (x0, y0), (x1, y1), (0, 0, 255), 8)

    return line_image


def lane_detection_pipeline(image, global_vals):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = global_vals.data.canny.low_threshold
    high_threshold = global_vals.data.canny.high_threshold

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    y = imshape[0]
    x = imshape[1]
    y_offset = 50
    x_offset = 20

    if global_vals.data.roi_vertices is None:
        global_vals.data.roi_vertices = np.array([[(15, y),
                                                   (x / 2 - x_offset, y / 2 + y_offset),
                                                   (x / 2 + x_offset, y / 2 + y_offset),
                                                   (x - 15, y)]],
                                                 dtype=np.int32)

    cv2.fillPoly(mask, global_vals.data.roi_vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, global_vals.data.hough.rho, global_vals.data.hough.theta, global_vals.data.hough.threshold, np.array([]),
                            global_vals.data.hough.min_line_length, global_vals.data.hough.max_line_gap)

    line_image = draw_hough_lines(lines, np.copy(image) * 0, global_vals)

    # Create a "color" binary image to combine with line image
    masked_edges = np.dstack((masked_edges, masked_edges, masked_edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return masked_edges, lines_edges

global_values = None
def set_global_values(gv):
    global global_values
    global_values = gv

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    global global_values
    masked_edges, result = lane_detection_pipeline(image, global_values)
    return result

def process_test_images():
    base_path = "test_images/"
    base_output_path = "output_images/"
    files = os.listdir(base_path)
    rows = len(files)
    for idx, file in enumerate(files):
        if idx <= 0:
            image = mpimg.imread(os.path.join(base_path, file))
            edge_img, line_overlay_img = lane_detection_pipeline(image)
            edge_img_filename = os.path.join(base_output_path, "edge_" + file)
            line_overlay_img_filename = os.path.join(base_output_path, "line_overlay_" + file)

            cv2.imwrite(edge_img_filename, edge_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(line_overlay_img_filename, line_overlay_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # Write the new file to the same test_images dir
            plt.subplot(rows, 3, idx * 3 + 1)
            plt.imshow(image)
            plt.subplot(rows, 3, idx * 3 + 2)
            plt.imshow(edge_img)
            plt.subplot(rows, 3, idx * 3 + 3)
            plt.imshow(line_overlay_img)

def process_single_image(dir, filename):
    global_extra_video_vals = Global(single_img_being_processed=True, use_dominant_line_approach=False)
    image = mpimg.imread(dir + "/" + filename)
    width = image.shape[1]
    height = image.shape[0]
    center_x = width / 2
    center_y = height / 2
    x_offset = 30
    y_offset = 240

    left_offset = 230
    bottom_offset = 55

    global_extra_video_vals.data.canny.low_threshold = 50
    global_extra_video_vals.data.canny.high_threshold = 220

    global_extra_video_vals.data.hough.rho = 1
    global_extra_video_vals.data.hough.theta = np.pi/180
    global_extra_video_vals.data.hough.threshold = 40
    global_extra_video_vals.data.hough.min_line_length = 30
    global_extra_video_vals.data.hough.max_line_gap = 200

    global_extra_video_vals.data.roi_vertices = np.array([[(left_offset, height - bottom_offset),
                                               (center_x - x_offset, center_y / 2 + y_offset),
                                               (center_x + x_offset, center_y / 2 + y_offset),
                                               (width - left_offset, height - bottom_offset)]],
                                             dtype=np.int32)
    image = mpimg.imread(os.path.join(dir, filename)) # "IMG_FROM_FRAME/frame625.jpg")
    edge_img, new_img = lane_detection_pipeline(image, global_extra_video_vals)
    cv2.imwrite("IMG_FROM_FRAME/experiment.jpg", new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite("IMG_FROM_FRAME/experiment_edge.jpg", edge_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

from moviepy.editor import VideoFileClip


def process_video_and_recreate(input_video_filename, output_video_filename):

    global_extra_video_vals = Global(single_img_being_processed=False, use_dominant_line_approach=False)

    clip1 = VideoFileClip(input_video_filename)
    width = clip1.w
    height = clip1.h
    center_x = width / 2
    center_y = height / 2
    x_offset = 30
    y_offset = 240

    left_offset = 230
    bottom_offset = 55

    global_extra_video_vals.data.canny.low_threshold = 50
    global_extra_video_vals.data.canny.high_threshold = 220

    global_extra_video_vals.data.hough.rho = 1
    global_extra_video_vals.data.hough.theta = np.pi/180
    global_extra_video_vals.data.hough.threshold = 40
    global_extra_video_vals.data.hough.min_line_length = 30
    global_extra_video_vals.data.hough.max_line_gap = 200

    global_extra_video_vals.data.roi_vertices = np.array([[(left_offset, height - bottom_offset),
                                               (center_x - x_offset, center_y / 2 + y_offset),
                                               (center_x + x_offset, center_y / 2 + y_offset),
                                               (width - left_offset, height - bottom_offset)]],
                                             dtype=np.int32)
    set_global_values(global_extra_video_vals)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_video_filename, audio=False)


import sys
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


def process_video_per_frame(input_video_filename):

    vidcap = cv2.VideoCapture(input_video_filename)
    success,image = vidcap.read()
    count = 0
    success = True
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    global_extra_video_vals = Global(single_img_being_processed=False, use_dominant_line_approach=False)

    center_x = width / 2
    center_y = height / 2
    x_offset = 30
    y_offset = 240

    left_offset = 200
    bottom_offset = 55

    global_extra_video_vals.data.canny.low_threshold = 40
    global_extra_video_vals.data.canny.high_threshold = 50


    global_extra_video_vals.data.roi_vertices = np.array([[(left_offset, height - bottom_offset),
                                               (center_x - x_offset, center_y / 2 + y_offset),
                                               (center_x + x_offset, center_y / 2 + y_offset),
                                               (width - left_offset, height - bottom_offset)]],
                                             dtype=np.int32)
    set_global_values(global_extra_video_vals)

    while success:
        success,image = vidcap.read()
        if not success:
          continue

        print_progress_bar(count, frame_count, prefix='Progress:', suffix='Complete', bar_length=50)
        cv2.imwrite("IMG_FROM_FRAME/frame%d.jpg" % count, image)     # save frame as JPEG file
        edge_img, new_img = lane_detection_pipeline(image, global_extra_video_vals)
        # cv2.imwrite("IMG_FROM_FRAME/edge_%d.jpg" % count, edge_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite("IMG_FROM_FRAME/overlay_%d.jpg" % count, new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        count += 1

    vidcap.release()
    cv2.destroyAllWindows()


def main():
    # process_video_per_frame("extra.mp4")
    # process_video_and_recreate("challenge.mp4", "extra.mp4")
    # process_video_per_frame("challenge.mp4")
    process_single_image("IMG_FROM_FRAME", "frame103.jpg")

if __name__ == "__main__":
    main()