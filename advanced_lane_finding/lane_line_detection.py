# The main file for Advanced Lane Line Detection
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
from experiment import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.style.use('ggplot')

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

DEBUG = False

class Utils:
    @staticmethod
    def read_image(filename: str):
        return mpimg.imread(filename)

    @staticmethod
    def display_images(imgs, is_grayscale, rows, cols, title, fig_size=(5, 5)):
        if len(imgs) != rows * cols:
            raise Exception("Invalid shape specified. Expected rows * cols = len(imgs)")

        per_plot_title = False
        if type(title) == list:
            if len(title) != len(imgs):
                raise Exception("Expected all images to have a title or a single title")
            per_plot_title = True

        fig, axs = plt.subplots(rows, cols, figsize=fig_size)
        canvas = FigureCanvas(fig)
        for r in range(rows):
            for c in range(cols):
                if is_grayscale:
                    if rows == 1:
                        axs[c].imshow(imgs[r * cols + c], cmap='gray')
                        axs[c].axes('off')
                        if per_plot_title:
                            axs[c].set_title(title[r * cols + c])
                    elif cols == 1:
                        axs[r].imshow(imgs[r * cols + c], cmap='gray')
                        axs[r].axis('off')
                        if per_plot_title:
                            axs[r].set_title(title[r * cols + c])
                    else:
                        axs[r, c].imshow(imgs[r*cols + c], cmap='gray')
                        axs[r].axis('off')
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

        if DEBUG:
            plt.show()

        canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        image.shape = (h, w, 3)
        return image

class ImageWrapper:
    def __init__(self, filename: str):
        self._distored_img = Utils.read_image(filename)
        self._undistorted_img = None
        self._grayscale_img = None
        self._roi_overlay = None
        self._perspective_transformed = None
        self._perspective_matrix = None
        self._inverse_matrix = None
        self.stages = []
        self.stages.append(self._distored_img)

    @property
    def undistorted(self):
        return self._undistorted_img

    @undistorted.setter
    def undistorted(self, img):
        self._undistorted_img = img
        self.stages.append(self.undistorted)
        self.grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @property
    def distorted(self):
        return self._distored_img

    @property
    def grayscale(self):
        return self._grayscale_img

    @grayscale.setter
    def grayscale(self, img):
        self._grayscale_img = img

    @property
    def roi_overlay(self):
        return self._roi_overlay

    @roi_overlay.setter
    def roi_overlay(self, img):
        self._roi_overlay = img
        self.stages.append(self._roi_overlay)

    @property
    def perspective_transform(self):
        return self._perspective_transformed

    @perspective_transform.setter
    def perspective_transform(self, img):
        self._perspective_transformed = img
        self.stages.append(img)

    def set_perspective_matrices(self, matrix, inv_matrix):
        self._perspective_matrix = matrix
        self._inverse_matrix = inv_matrix

    @property
    def shape(self):
        if self._undistorted_img is None:
            return (self._distored_img.shape[1], self._distored_img.shape[0])
        else:
            return (self._undistorted_img.shape[1], self._undistorted_img.shape[0])

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

                    if DEBUG:
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

    def unwarp(self, img : ImageWrapper, display_img = False):
        if self.camera_calibrated_:
            dst = cv2.undistort(img.distorted, self.mtx, self.dist, None, self.mtx)

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
    @staticmethod
    def _define_source_polygon(img):
        img_size = img.shape
        source_transformation = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

        return source_transformation

    @staticmethod
    def _define_destination_polygon(img):
        img_size = img.shape
        destination_transformation = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

        return destination_transformation

    @staticmethod
    def warp_perspective_to_top_down(img : ImageWrapper):
        """
        Calculate the inversed transformation matrix and warped top down transform image
        :param src: source points where warp transforms from
        :param img: stacked binary thresholded image that includes saturation, gradient direction, colour intensity
        :param dst: destination points where warp transforms to
        :return: the transformation matrix inversed, warped top-down binary image
        """
        img_size = img.shape
        src = RegionOfInterest._define_source_polygon(img)
        dst = RegionOfInterest._define_destination_polygon(img)
        transformation_matrix = cv2.getPerspectiveTransform(src, dst)  # the transform matrix
        transformation_matrix_inverse = cv2.getPerspectiveTransform(dst, src)  # the transform matrix inverse
        perspective_top_down = cv2.warpPerspective(img.undistorted, transformation_matrix, (img_size))  # warp image to a top-down view
        if DEBUG:
            plt.title("Perspective Top Down")
            plt.imshow(perspective_top_down)
            plt.show()

        return transformation_matrix, transformation_matrix_inverse, perspective_top_down

    @staticmethod
    def polygon_overlay_img(img : ImageWrapper):
        """
        Create green polygon and overlay on undistorted image.
        """
        src = RegionOfInterest._define_source_polygon(img)
        dst = RegionOfInterest._define_destination_polygon(img)
        polygon_src_overlay_img = np.copy(img.undistorted)
        polygon_src_undistored_image = cv2.line(polygon_src_overlay_img, tuple(src[0]), tuple(src[1]), [0, 255, 0], 3)
        polygon_src_undistored_image = cv2.line(polygon_src_undistored_image, tuple(src[1]), tuple(src[2]), [0, 255, 0], 3)
        polygon_src_undistored_image = cv2.line(polygon_src_undistored_image, tuple(src[2]), tuple(src[3]), [0, 255, 0], 3)
        polygon_src_undistored_image = cv2.line(polygon_src_undistored_image, tuple(src[3]), tuple(src[0]), [0, 255, 0], 3)

        polygon_dst_overlay_img = np.copy(img.undistorted)
        polygon_dst_undistored_image = cv2.line(polygon_dst_overlay_img, tuple(dst[0]), tuple(dst[1]), [0, 255, 0], 3)
        polygon_dst_undistored_image = cv2.line(polygon_dst_undistored_image, tuple(dst[1]), tuple(dst[2]), [0, 255, 0], 3)
        polygon_dst_undistored_image = cv2.line(polygon_dst_undistored_image, tuple(dst[2]), tuple(dst[3]), [0, 255, 0], 3)
        polygon_dst_undistored_image = cv2.line(polygon_dst_undistored_image, tuple(dst[3]), tuple(dst[0]), [0, 255, 0], 3)

        if DEBUG:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(polygon_src_undistored_image)
            ax2.imshow(polygon_dst_undistored_image)
            ax1.set_title("Src Region of Interest defined")
            ax2.set_title("Dst Region of Interest defined")
            plt.show()

        return polygon_src_undistored_image, polygon_dst_undistored_image

class Thresholder:
    @staticmethod
    def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dx, dy = (1, 0) if orient == 'x' else (0, 1)
        sobel = cv2.Sobel(grayscale, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    @staticmethod
    def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(abs_sobelxy)
        abs_sobelxy = (abs_sobelxy / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(abs_sobelxy)
        binary_output[(abs_sobelxy >= mag_thresh[0]) & (abs_sobelxy <= mag_thresh[1])] = 1
        return binary_output

    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        # Apply threshold
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        dir_gradient = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(dir_gradient)
        binary_output[(dir_gradient >= thresh[0]) & (dir_gradient <= thresh[1])] = 1
        return binary_output

def pipeline(imgs, args):
    cameracalib = CameraCalibration(chessboard_size=(9, 6), args=args)
    options = ExperimentManagerOptions()
    options.overwrite_if_experiment_exists = True
    experiment_manager = ExperimentManager(options)
    experiment = experiment_manager.new_experiment(args.experiment)

    for i, img in enumerate(imgs):
        logger.info("Performing Distortion Correction on Image: %s" % i)
        unwarped = cameracalib.unwarp(img, False)

        distorted_undistored_img = Utils.display_images([img.distorted, unwarped], False, 1, 2, ["Distorted", "Undistorted"], fig_size=(15, 5))

        experiment.add_image(distorted_undistored_img, "output", "distorted_undistorted_" + str(i) + ".png",
                             "Unwarped Image from Camera Calibration",
                             description="This is the image obtained from the camera calibration")

        img.undistorted = unwarped
        roi_img = RegionOfInterest.polygon_overlay_img(img)
        img.roi_overlay = roi_img
        transformation_matrix, inv_transformation_matrix, perspective_transformed_img = RegionOfInterest.warp_perspective_to_top_down(img)
        img.perspective_transform = perspective_transformed_img
        img.set_perspective_matrices(transformation_matrix, inv_transformation_matrix)
        if DEBUG:
            plt.imshow(img.perspective_transform)
            plt.show()

    experiment_manager.to_markdown(args.experiment, args.experiment + ".md")

def test_images_pipeline():
    test_files = os.listdir("test_images")
    test_files = ["test_images/" + f for f in test_files]

    test_imgs = []
    for file in test_files:
        img = ImageWrapper(file)
        test_imgs.append(img)

    return test_imgs

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", help="The pipeline to use.", type=str, choices=["test, video"], default="test")
    parser.add_argument("--experiment", help="The experiment name to try", type=str)

    args = parser.parse_args()
    return args

def main():
    args = argument_parser()

    if args.pipeline == "test":
        imgs = test_images_pipeline()
        pipeline(imgs, args)

if __name__ == "__main__":
    main()