# The main file for Advanced Lane Line Detection
# python lane_line_detection.py --pipeline video --file challenge_video.mp4
# python lane_line_detection.py --pipeline test
# python lane_line_detection.py --pipeline image --file test_images/straight_lines1.jpg
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import cv2
import pickle
import logging
import typing
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
# from experiment import *
from moviepy.editor import VideoFileClip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.style.use('ggplot')

ARGS = None
#################################### Setup Logging ########################################
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
logger.setLevel(logging.ERROR)

class Utils:
    @staticmethod
    def binary_to_3_channel_img(binary_img):
        new_img = np.dstack((binary_img, binary_img, binary_img)) * 255
        return new_img

    @staticmethod
    def plot_binary(binary_img):
        new_img = Utils.binary_to_3_channel_img(binary_img)
        plt.imshow(new_img)

    @staticmethod
    def read_image(filename: str):
        return mpimg.imread(filename)
        # return cv2.imread(filename)

    @staticmethod
    def write_image(filename, img):
        mpimg.imsave(filename, img)

    @staticmethod
    def matplotlib_row_col_display_imgs(output_name, imgs, is_grayscale, rows, cols, title, fig_size=(5, 5)):
        if len(imgs) != rows * cols:
            raise Exception("Invalid shape specified. Expected rows * cols = len(imgs)")

        per_plot_title = False
        if type(title) == list:
            if len(title) != len(imgs):
                raise Exception("Expected all images to have a title or a single title")
            per_plot_title = True

        fig, axs = plt.subplots(rows, cols, figsize=fig_size)
        # canvas = FigureCanvas(fig)
        for r in range(rows):
            for c in range(cols):
                if is_grayscale:
                    if rows == 1:
                        axs[c].imshow(imgs[r * cols + c], cmap='gray')
                        axs[c].axis('off')
                        if per_plot_title:
                            axs[c].set_title(title[r * cols + c])
                    elif cols == 1:
                        axs[r].imshow(imgs[r * cols + c], cmap='gray')
                        axs[r].axis('off')
                        if per_plot_title:
                            axs[r].set_title(title[r * cols + c])
                    else:
                        axs[r, c].imshow(imgs[r*cols + c], cmap='gray')
                        axs[r, c].axis('off')
                        if per_plot_title:
                            axs[r, c].set_title(title[r * cols + c])
                else:
                    if rows == 1:
                        axs[c].imshow(imgs[r * cols + c])
                        axs[c].axis('off')
                        if per_plot_title:
                            axs[c].set_title(title[r * cols + c])
                    elif cols == 1:
                        axs[r].imshow(imgs[r * cols + c])
                        axs[r].axis('off')
                        if per_plot_title:
                            axs[r].set_title(title[r * cols + c])
                    else:
                        axs[r, c].imshow(imgs[r*cols + c])
                        axs[r, c].axis('off')
                        if per_plot_title:
                            axs[r, c].set_title(title[r*cols + c])

        if not per_plot_title:
            plt.title(title)

        # canvas.draw()
        # w, h = fig.canvas.get_width_height()
        # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        # image.shape = (h, w, 3)
        # return image

        plt.savefig(output_name)

    @staticmethod
    def merge_images(*imgs):
        new_im = np.concatenate(imgs, axis=1)
        return new_im

    @staticmethod
    def merge_2x2(img1, img2, img3, img4):
        new_img1 = np.concatenate([img1, img2], axis=1)
        new_img2 = np.concatenate([img3, img4], axis=1)
        combined = np.concatenate([new_img1, new_img2], axis=0)
        return combined

    @staticmethod
    def overlay_text(image, text, pos=(0, 0), color=(255, 255, 255)):
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("./fonts/liberation-sans.ttf", 64)
        draw.text(pos, text, color, font=font)
        image = np.asarray(image)

        return image

    @staticmethod
    def create_report_images(output_name, img1, img2, title1, title2, grayscale=False):
        output_name = "report/" + output_name
        # Try out a chess board to make sure that the calibration works
        Utils.matplotlib_row_col_display_imgs(output_name,
                                             [img1, img2],
                                             is_grayscale=grayscale,
                                             rows = 1, cols = 2,
                                             title=[title1, title2],
                                             fig_size=(10, 5))

    @staticmethod
    def create_report_image(output_name, img, title):
        output_name = "report/" + output_name
        plt.figure()
        plt.axis('off')
        plt.title(title)
        plt.imshow(img)
        plt.savefig(output_name)

class CameraCalibration:
    def __init__(self, chessboard_size, args):
        self.camera_calibrated_ = False
        mtx, dist = self.load_calibration_()
        if mtx is None and dist is None:
            logger.info("Calibrating the Camera")
            calib_files = os.listdir("camera_cal")
            calib_files = ["camera_cal/" + f for f in calib_files]

            objpoints = []
            imgpoints = []

            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

            imgshape = None
            for calib in calib_files:
                calib_img = Utils.read_image(calib)
                gray  = cv2.cvtColor(calib_img, cv2.COLOR_RGB2GRAY)
                imgshape = gray.shape

                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if ret:
                    imgpoints.append(corners)
                    objpoints.append(objp)

                    if args.debug:
                        chessboard_corners_img = cv2.drawChessboardCorners(calib_img, chessboard_size, corners, ret)
                        plt.imshow(chessboard_corners_img)
                        plt.show()

            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape, None, None)
            self.save_calibration()
        else:
            logger.info("Camera Calibration loaded from file: calibration.p")
            self.mtx = mtx
            self.dist = dist

        self.camera_calibrated_ = True

    def load_calibration_(self):
        mtx, dist = None, None
        if os.path.exists('calibration.p'):
            f = open('calibration.p', 'rb')
            p = pickle.load(f)
            mtx = p['mtx']
            dist = p['dist']

        return mtx, dist

    def save_calibration(self):
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open("calibration.p", "wb"))

    def unwarp(self, img, display_img = False):
        if self.camera_calibrated_:
            dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

            if display_img:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.set_title("Warped")
                ax1.imshow(img)
                ax1.axis('off')
                ax2.set_title("Unwarped")
                ax2.imshow(dst)
                ax2.axis('off')
                plt.show()

            return dst

        logger.error("Unwarp can only be called when camera is calibrated")
        raise Exception("Unwarp can only be called when camera is calibrated")

class RegionOfInterest:
    def __init__(self):
        self.transformation_matrix = None
        self.inverse_matrix = None

    @staticmethod
    def _define_source_polygon(img):
        img_size = (img.shape[1], img.shape[0])
        source_transformation = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

        return source_transformation

    @staticmethod
    def _define_destination_polygon(img):
        img_size = (img.shape[1], img.shape[0])
        destination_transformation = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

        return destination_transformation

    def warp_perspective_to_top_down(self, img):
        """
        Calculate the inversed transformation matrix and warped top down transform image
        :param src: source points where warp transforms from
        :param img: stacked binary thresholded image that includes saturation, gradient direction, colour intensity
        :param dst: destination points where warp transforms to
        :return: the transformation matrix inversed, warped top-down binary image
        """
        img_size = (img.shape[1], img.shape[0])
        src = RegionOfInterest._define_source_polygon(img)
        dst = RegionOfInterest._define_destination_polygon(img)
        transformation_matrix = cv2.getPerspectiveTransform(src, dst)  # the transform matrix
        transformation_matrix_inverse = cv2.getPerspectiveTransform(dst, src)  # the transform matrix inverse
        perspective_top_down = cv2.warpPerspective(img, transformation_matrix, (img_size))  # warp image to a top-down view

        self.transformation_matrix = transformation_matrix
        self.inverse_matrix = transformation_matrix_inverse
        return perspective_top_down

    def top_down_to_original(self, img, size_img):
        return cv2.warpPerspective(img, self.inverse_matrix, tuple(list(size_img.shape[1::-1])))

    @staticmethod
    def polygon_overlay_img(img):
        global ARGS
        """
        Create green polygon and overlay on undistorted image.
        """
        src = RegionOfInterest._define_source_polygon(img)
        dst = RegionOfInterest._define_destination_polygon(img)
        polygon_src_overlay_img = np.copy(img)
        src_transformation_img = cv2.line(polygon_src_overlay_img, tuple(src[0]), tuple(src[1]), [0, 255, 0], 3)
        src_transformation_img = cv2.line(src_transformation_img, tuple(src[1]), tuple(src[2]), [0, 255, 0], 3)
        src_transformation_img = cv2.line(src_transformation_img, tuple(src[2]), tuple(src[3]), [0, 255, 0], 3)
        src_transformation_img = cv2.line(src_transformation_img, tuple(src[3]), tuple(src[0]), [0, 255, 0], 3)

        polygon_dst_overlay_img = np.copy(img)
        dst_transformation_img = cv2.line(polygon_dst_overlay_img, tuple(dst[0]), tuple(dst[1]), [0, 255, 0], 3)
        dst_transformation_img = cv2.line(dst_transformation_img, tuple(dst[1]), tuple(dst[2]), [0, 255, 0], 3)
        dst_transformation_img = cv2.line(dst_transformation_img, tuple(dst[2]), tuple(dst[3]), [0, 255, 0], 3)
        dst_transformation_img = cv2.line(dst_transformation_img, tuple(dst[3]), tuple(dst[0]), [0, 255, 0], 3)

        if False:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(src_transformation_img)
            ax2.imshow(dst_transformation_img)
            ax1.set_title("Src Region of Interest defined")
            ax2.set_title("Dst Region of Interest defined")
            plt.show()

        return src_transformation_img, dst_transformation_img

class Thresholder:
    @staticmethod
    def simple_threshold(img, thresh=(0, 255)):
        binary = np.zeros_like(img)
        binary[(img > thresh[0]) & (img <= thresh[1])] = 1
        return binary

    @staticmethod
    def abs_sobel_threshold(img, perform_grayscaling, orient='x', sobel_kernel=3, thresh=(0, 255)):
        if perform_grayscaling:
            grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = img
        dx, dy = (1, 0) if orient == 'x' else (0, 1)
        sobel = cv2.Sobel(grayscale, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    @staticmethod
    def mag_thresh(img, perform_grayscaling, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        if perform_grayscaling:
            grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = img
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(abs_sobelxy)
        abs_sobelxy = (abs_sobelxy / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(abs_sobelxy)
        binary_output[(abs_sobelxy >= mag_thresh[0]) & (abs_sobelxy <= mag_thresh[1])] = 1
        return binary_output

    @staticmethod
    def dir_threshold(img, perform_grayscaling, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        # Apply threshold
        if perform_grayscaling:
            grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = img
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        dir_gradient = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(dir_gradient)
        binary_output[(dir_gradient >= thresh[0]) & (dir_gradient <= thresh[1])] = 1
        return binary_output

    @staticmethod
    def combined_filters(img, perform_grayscaling, sobel_thresh : tuple, mag_thresh : tuple, dir_thresh : tuple):
        if perform_grayscaling:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ksize = 3
        gradx = Thresholder.abs_sobel_threshold(img, perform_grayscaling=False, orient='x', sobel_kernel=ksize, thresh=sobel_thresh)
        grady = Thresholder.abs_sobel_threshold(img, perform_grayscaling=False, orient='y', sobel_kernel=ksize, thresh=sobel_thresh)
        mag_binary = Thresholder.mag_thresh(img, perform_grayscaling=False, sobel_kernel=ksize, mag_thresh=mag_thresh)
        dir_binary = Thresholder.dir_threshold(img, perform_grayscaling=False, sobel_kernel=ksize, thresh=dir_thresh)

        combined = np.zeros_like(img)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

    @staticmethod
    def color_threshold(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))

        sensitivity_1 = 68
        white = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

        sensitivity_2 = 60
        hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        white_2 = cv2.inRange(hsl, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
        white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

        # s_binary = np.zeros_like(img)
        # return s_binary[(yellow == 1) | (white_2 == 1) | (white_3 == 1)]
        return (yellow | white | white_2 | white_3) // 255

def overlay_images(img1, img2):
    assert img1.shape == img2.shape
    overlay_img = np.zeros_like(img1)

    overlay_img[(img1 == 1) | (img2 == 1)] = 1
    return overlay_img

def get_saturation_channel(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    return s

# Values from Udacity
X_METER_PER_PIXEL = 3.7/700
Y_METER_PER_PIXEL = 30/720

class Lane:
    def __init__(self, xs, ys, height_to_generate):
        self._xs_pixels = xs
        self._ys_pixels = ys
        # Height is not required but just because its easy am storing it.
        self._height = height_to_generate

        self._xs = xs * X_METER_PER_PIXEL
        self._ys = ys * Y_METER_PER_PIXEL

    def fit(self):
        # Fit a 2nd order polynomial
        if self._xs is not None and self._ys is not None and len(self._xs) > 0 and len(self._ys) > 0:
            return np.polyfit(self._xs, self._ys, 2)

        return None

    def pixels_fit(self):
        if self._xs_pixels is not None and self._ys_pixels is not None and len(self._xs_pixels) > 0 and len(self._ys_pixels) > 0:
            return np.polyfit(self._xs_pixels, self._ys_pixels, 2)

        return None

    def p(self):
        return np.poly1d(self.fit())

    def p1(self):
        # First Derivative
        return np.polyder(self.p())

    def p2(self):
        # 2nd Derivative
        return np.polyder(self.p(), 2)

    def _curvature(self, y):
        scaled_y = y * Y_METER_PER_PIXEL
        pixels_fit = self.pixels_fit()
        if pixels_fit is None:
            return None

        A = pixels_fit[0]
        B = pixels_fit[1]
        c = ((1 + (2 * A * scaled_y + B) ** 2) ** 1.5) / np.absolute(2 * A)
        return c

    def curvature(self):
        return self._curvature(self._height)

    def get_lane_fit(self):
        try:
            ploty = np.linspace(0, self._height - 1, self._height)
            pixel_fit = self.pixels_fit()
            line_fit = pixel_fit[0] * ploty ** 2 + self.pixels_fit()[1] * ploty + pixel_fit[2]
        except:
            raise Exception("Failed during lane fitting")

        return line_fit, ploty

class Frame:
    def __init__(self, leftx, lefty, rightx, righty):
        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.righty = righty

class LaneFinder:
    def __init__(self, num_frames):
        self._left_fit = None
        self._right_fit = None
        self._right_lanex = None
        self._right_laney = None
        self._left_laney = None
        self._left_lanex = None
        # Set the width of the windows +/- margin
        self._margin = 100
        self.num_frame_avg = num_frames
        self.previous_frames = []

    @property
    def left_fit(self):
        return self._left_fit

    @property
    def right_fit(self):
        return self._right_fit

    def full_lane_finding_step(self, binary_img):
        Utils.plot_binary(binary_img)
        height = binary_img.shape[0]

        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)

        # Create the output image to draw on
        out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

        # Find the peak from the left and right halves of the image.
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = midpoint + np.argmax(histogram[midpoint:])

        # Number of Sliding Windows
        nwindows = 9

        # Set the height of the windows
        window_height = height // nwindows

        # Identify the x and y positions of all non-zero pixels
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_idxs = []
        right_lane_idxs = []

        # Step through the windows
        i = 0
        centers_left_x = []
        centers_left_y = []
        centers_right_x = []
        centers_right_y = []

        for window in range(nwindows):
            # Identify window boundaries in x & y ( and right and left )
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            win_xleft_low = leftx_current - self._margin
            win_xleft_high = leftx_current + self._margin

            win_xright_low = rightx_current - self._margin
            win_xright_high = rightx_current + self._margin

            # Draw the windows
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x & y within the window
            good_left_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_idxs.append(good_left_idxs)
            right_lane_idxs.append(good_right_idxs)

            if len(good_left_idxs) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_idxs]))

            if len(good_right_idxs) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_idxs]))

            # Constrain the polyline fits to the center of the boxes. If they are not correct then we are in the
            # wrong region looking for them
            center_left_x = (win_xleft_low + win_xleft_high) // 2
            center_right_x = (win_xright_low + win_xright_high) // 2
            center_y = (win_y_low + win_y_high) // 2

            centers_left_x.append(center_left_x)
            centers_left_y.append(center_y)
            centers_right_x.append(center_right_x)
            centers_right_y.append(center_y)

        centers_left_x = np.array(centers_left_x)
        centers_left_y = np.array(centers_left_y)
        centers_right_x = np.array(centers_right_x)
        centers_right_y = np.array(centers_right_y)

        # Concatenate the array of indices
        left_lane_idxs = np.concatenate(left_lane_idxs)
        right_lane_idxs = np.concatenate(right_lane_idxs)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_idxs]
        lefty = nonzeroy[left_lane_idxs]
        rightx = nonzerox[right_lane_idxs]
        righty = nonzeroy[right_lane_idxs]

        leftx = np.concatenate((leftx, centers_left_x[-1:-2]))
        lefty = np.concatenate((lefty, centers_left_y[-1:-2]))
        rightx = np.concatenate((rightx, centers_right_x[-1:-2]))
        righty = np.concatenate((righty, centers_right_y[-1:-2]))

        if len(self.previous_frames) > self.num_frame_avg:
            self.previous_frames.pop()

        # self.current_left_lane = Lane(leftx, lefty, height)
        # self.current_right_lane = Lane(rightx, righty, height)
        frame = Frame(leftx, lefty, rightx, righty)

        # Now left average the frames
        for frame in self.previous_frames:
            leftx = np.concatenate((leftx, frame.leftx))
            lefty = np.concatenate((lefty, frame.lefty))
            rightx = np.concatenate((rightx, frame.rightx))
            righty = np.concatenate((righty, frame.righty))

        # Fit a second order polynomial to each
        left_fit = None
        if len(leftx) > 0 and len(lefty) >  0:
            left_fit = np.polyfit(lefty, leftx, 2)

        right_fit = None
        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        # Insert the current frame
        self.previous_frames.insert(0, frame)

        # Generate x and y values for plotting
        ploty = np.linspace(0, height - 1, height)

        left_fitx = None
        if left_fit is not None:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        right_fitx = None
        if right_fit is not None:
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_idxs], nonzerox[left_lane_idxs]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_idxs], nonzerox[right_lane_idxs]] = [0, 0, 255]

        if left_fitx is not None:
            plt.plot(left_fitx, ploty, color='yellow')

        if right_fitx is not None:
            plt.plot(right_fitx, ploty, color='yellow')

        self._left_fit = left_fit
        self._right_fit = right_fit
        self._left_lanex = left_fitx
        self._left_laney = ploty
        self._right_lanex = right_fitx
        self._right_laney = ploty

        return out_img

    def partial_lane_finding_step(self, binary_img):
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = self._left_fit
        right_fit = self._right_fit

        margin = 100

        left_lane_idxs = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_idxs = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_idxs]
        lefty = nonzeroy[left_lane_idxs]
        rightx = nonzerox[right_lane_idxs]
        righty = nonzeroy[right_lane_idxs]

        # Fit a second order polynomial to each
        left_fit = None
        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)

        right_fit = None
        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        self._left_fit = left_fit
        self._right_fit = right_fit
        self._current_img = binary_img


        if left_fit is not None and right_fit is not None:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            self._left_lanex = left_fitx
            self._right_lanex = right_fitx

    def visualize_lane_fits(self, inp_img):
        # Generate x and y values for plotting
        out_img = np.copy(inp_img)

        ploty = np.linspace(0, inp_img.shape[0] - 1, inp_img.shape[0])
        if self._left_fit is not None:
            left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
            pts1 = (np.dstack([left_fitx, ploty]))[0].astype(np.int32)
            cv2.polylines(out_img, [pts1], False, (255, 255, 0), 3)

        if self._right_fit is not None:
            right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
            pts2 = (np.dstack([right_fitx, ploty]))[0].astype(np.int32)
            cv2.polylines(out_img, [pts2], False, (255, 255, 0), 3)

        return out_img

    def overlay_lane(self, image, left_lane, right_lane, transform):
        left_ys = np.linspace(0, 100, num=101) * 7.2

        left_fit = left_lane
        right_fit = right_lane

        left_xs = None
        if left_fit is not None:
            left_xs = left_fit[0] * left_ys ** 2 + left_fit[1] * left_ys + left_fit[2]

        right_ys = np.linspace(0, 100, num=101) * 7.2

        right_xs = None
        if right_fit is not None:
            right_xs = right_fit[0] * right_ys ** 2 + right_fit[1] * right_ys + right_fit[2]

        color_warp = np.zeros_like(image).astype(np.uint8)

        pts_left, pts_right = [], []

        if left_xs is not None:
            pts_left = np.array([np.transpose(np.vstack([left_xs, left_ys]))])

        if right_xs is not None:
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xs, right_ys])))])

        new_img = image
        if len(pts_left) == len(pts_right) and len(pts_left) > 0:
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            newwarp = cv2.warpPerspective(color_warp, transform, image.shape[1::-1])

            new_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        return new_img

    def compute_curvature(self, lanex, height):
        if self._left_fit is None or self._right_fit is None:
            return 0.0

        scaled_height = height * Y_METER_PER_PIXEL
        ploty = np.linspace(0, height - 1, num=height)# to cover same y-range as image

        fit_cr = np.polyfit(ploty * Y_METER_PER_PIXEL, lanex * X_METER_PER_PIXEL, 2)

        A = fit_cr[0]
        B = fit_cr[1]
        c = ((1 + (2 * A * scaled_height + B) ** 2) ** 1.5) / np.absolute(2 * A)
        return c

    def distance_from_center(self, center):
        TO_METER = np.array([[X_METER_PER_PIXEL, 0],
                             [0, Y_METER_PER_PIXEL]])

        center_dot = np.dot(center, TO_METER)

        if self._left_fit is None or self._right_fit is None:
            return 0.0

        left_x = np.poly1d(np.polyfit(self._left_laney * Y_METER_PER_PIXEL, self._left_lanex * X_METER_PER_PIXEL, 2))(center_dot[1])
        right_x  = np.poly1d(np.polyfit(self._right_laney * Y_METER_PER_PIXEL, self._right_lanex * X_METER_PER_PIXEL, 2))(center_dot[1])

        return ((right_x + left_x) / 2 - center_dot[0])

    def overlay(self, img, roi : RegionOfInterest):
        shape = img.shape
        # left_curvature = left_lane.curvature()
        # right_curvature = right_lane.curvature()

        if self._left_fit is None:
            left_curvature = 0.0
        else:
            left_curvature = self.compute_curvature(self._left_lanex, shape[0])

        if self._right_fit is None:
            right_curvature = 0.0
        else:
            right_curvature = self.compute_curvature(self._right_lanex, shape[0])

        center_point = (shape[1] / 2, shape[0])
        center_distance = self.distance_from_center(center_point)

        img = self.overlay_lane(img, self._left_fit, self._right_fit, roi.inverse_matrix)

        left_overlay = "Left curvature: {0:.2f}m".format(left_curvature)
        img = Utils.overlay_text(img, left_overlay, pos=(10, 10))

        right_overlay = "Right curvature: {0:.2f}m".format(right_curvature)
        img = Utils.overlay_text(img, right_overlay, pos=(10, 90))

        center_overlay = "Distance from center: {0:.2f}m".format(center_distance)
        img = Utils.overlay_text(img, center_overlay, pos=(10, 180))

        return img


lane_finder = LaneFinder(num_frames=5)
roi = RegionOfInterest()

counter = 0

def pipeline(img):
    global ARGS, counter
    lane_finder.num_frame_avg = ARGS.frames
    filename = "failed/" + str(counter) + ".png"
    cv2.imwrite(filename, img)
    cameracalib = CameraCalibration(chessboard_size=(9, 6), args=ARGS)

    # Undistort the image
    distorted = img

    if ARGS.report:
        # Try out a chess board to make sure that the calibration works
        chess_board_distorted = Utils.read_image('camera_cal/calibration1.jpg')
        unwarped_chess_board = cameracalib.unwarp(chess_board_distorted, display_img=False)
        Utils.create_report_images("chess_board_undistorted.png",
                                   chess_board_distorted,
                                   unwarped_chess_board,
                                   "Distorted Chess Board",
                                   "Undistorted Chess Board")

    logger.info("Performing Distortion Correction on Image:")
    unwarped = cameracalib.unwarp(distorted, display_img=False)

    if ARGS.report:
        Utils.create_report_images("input_image_undistorted.png", distorted, unwarped, "Distorted Image", "Undistorted Image")

    saturation_channel_img = get_saturation_channel(unwarped)
    binary = Thresholder.simple_threshold(saturation_channel_img, thresh=(90, 255))
    if ARGS.debug:
        plt.imshow(binary, cmap='gray')
        plt.title("Saturation Image")
        plt.show()

    if ARGS.report:
        Utils.create_report_images("saturation_threshold.png",
                                   saturation_channel_img,
                                   Utils.binary_to_3_channel_img(binary),
                                   "Saturation Channel",
                                   "Thresholded Saturation Channel",
                                   grayscale=True)
    #
    # # Perform the Thresholding on the image
    # thresholded = Thresholder.combined_filters(unwarped, True, (10, 100), (200, 255), (0.0, 0.6))
    #
    # if DEBUG:
    #     plt.imshow(thresholded, cmap='gray')
    #     plt.title("Thresholded Image")
    #     plt.show()

    # overlay_img = overlay_images(binary, thresholded)
    color_thresholded_img = Thresholder.color_threshold(unwarped)
    if ARGS.debug:
        plt.imshow(color_thresholded_img, cmap='gray')
        plt.title("Overlay Image")
        plt.show()

    overlay_img = overlay_images(binary, color_thresholded_img)
    overlay_img = color_thresholded_img

    if ARGS.report:
        Utils.create_report_images("merged_thresholding.png",
                                   Utils.binary_to_3_channel_img(color_thresholded_img),
                                   Utils.binary_to_3_channel_img(overlay_img),
                                   "Color Thresholded Image",
                                   "Merged Final Thresholded Image")

    if ARGS.debug:
        plt.imshow(overlay_img, cmap='gray')
        plt.title("Overlay Image")
        plt.show()

    src_transformation_img, dst_transformation_img = RegionOfInterest.polygon_overlay_img(img)
    if ARGS.report:
        Utils.create_report_images("roi.png",
                                   src_transformation_img,
                                   dst_transformation_img,
                                   "Src Transformation",
                                   "Dst Transformation")


    perspective_transformed_img = roi.warp_perspective_to_top_down(overlay_img)
    if ARGS.debug:
        plt.title("Perspective Transformed Image")
        Utils.plot_binary(perspective_transformed_img)

    if ARGS.report:
        Utils.create_report_image("perspective.png",
                                  Utils.binary_to_3_channel_img(perspective_transformed_img),
                                  "Perspective Transformed Image")

    # plt.imshow(perspective_transformed_img, cmap='gray')
    # plt.title("Perspective Transformed Image")
    # plt.show()

    if lane_finder.left_fit is None or lane_finder.right_fit is None or counter < 40 or ARGS.use_full_detection:
        lane_finder_img = lane_finder.full_lane_finding_step(perspective_transformed_img)
        lane_finder_img = lane_finder.visualize_lane_fits(lane_finder_img)
    else:
        lane_finder.partial_lane_finding_step(perspective_transformed_img)
        if lane_finder.left_fit is None or lane_finder.right_fit is None:
            lane_finder_img = lane_finder.full_lane_finding_step(perspective_transformed_img)
        else:
            lane_finder_img = lane_finder.visualize_lane_fits(Utils.binary_to_3_channel_img(perspective_transformed_img))

    if ARGS.report:
        Utils.create_report_image("lane_finder_algo.png", lane_finder_img, "Lane Finding Algorithm")

    # if DEBUG:
    #     plt.title("Lane Overlaid")
    #     Utils.plot_binary(binary_img)
    final_img = lane_finder.overlay(unwarped, roi)

    if ARGS.report:
        Utils.create_report_image("final_image.png", final_img, "Lane Overlaid Image")


    if ARGS.debug:
        plt.title("Final Image")
        plt.imshow(final_img)
        plt.axis('off')
        plt.show()

    # frame = "Frame: %s" % counter
    # final_img = Utils.overlay_text(final_img, frame, pos=(10, 10))

    # plt.imshow(final_img)
    # plt.show()

    all =  Utils.merge_2x2(unwarped, Utils.binary_to_3_channel_img(overlay_img), lane_finder_img, final_img)
    # os.remove(filename)
    counter += 1
    cv2.imwrite("failed/all_" + str(counter) + ".png", all)
    # return final_img
    return all

def test_images_pipeline():
    test_files = os.listdir("test_images")
    test_files = ["test_images/" + f for f in test_files]

    test_imgs = []
    for file in test_files:
        img = Utils.read_image(file)
        test_imgs.append(img)

    return test_imgs

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", help="The pipeline to use.", type=str, choices=["test", "image", "video"], default="test")
    parser.add_argument("--experiment", help="The experiment name to try", type=str)
    parser.add_argument("--file", help="Video / Image file to use", type=str)
    parser.add_argument("--debug", help="Debug experiment", action="store_true")
    parser.add_argument("--report", help="Should we save the images for the report", action="store_true")
    parser.add_argument("--use_full_detection", help="Use the full lane line detection algorithm", action="store_true")
    parser.add_argument("--frames", help="Number of frames to average", type=int, default=5)


    args = parser.parse_args()
    return args

def main():
    global ARGS
    args = argument_parser()

    ARGS = args
    if args.pipeline == "test":
        imgs = test_images_pipeline()
        for img in imgs:
            pipeline(img)
    elif args.pipeline == "image":
        img = Utils.read_image(args.file)
        result = pipeline(img)
        # plt.figure(figsize=(25, 10))
        # plt.axis('off')
        # plt.imshow(result)
        # plt.show()
    elif args.pipeline == "video":
        clip = VideoFileClip(args.file)
        output_clip = clip.fl_image(pipeline)
        output_clip.write_videofile("output_" + args.file, audio=False)

if __name__ == "__main__":
    main()
