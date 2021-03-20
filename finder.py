from functools import reduce, partial
from queue import LifoQueue, Queue
from multiprocessing.managers import BaseManager
import numpy as np
import datetime
from math import tan, radians, degrees, sqrt, sin, atan
from typing import List, Tuple, Union, Callable, Dict

from imutils.video import VideoStream
import cv2


class PowercellFinder:
    # HSV color boundaries for detecting the powercells
    # DEFAULT_BOUNDS = [(24, 121, 190, 78, 247, 247), (66, 234, 209, 81, 243, 243), (27, 27, 218, 79, 245, 255), (20, 156, 83, 60, 253, 220)]  # v6
    DEFAULT_BOUNDS = [(26, 109, 104, 31, 255, 255), (30, 56, 185, 33, 213, 255)]
    DEFAULT_PREPROCESS = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV)
    DEFAULT_POSTPROCESS = (lambda *args: None)

    def __init__(self, src: int, preprocess_fn=None, fov=60, height=0.6, offset_angle=-20, ball_area_thresh=0.6, bounds=None, ball_radius=0.09):
        self.src = src  # camera source number
        self.preprocess_fn = preprocess_fn  # Function to run on a frame before finding powercells
        self.fov = fov  # diagonal FOV of camera
        self.height = height - ball_radius  # height of camera off the center of the balls (on the ground)
        self.offset_angle = offset_angle  # angle the top of the FOV is from the horizontal
        self.ball_area_thresh = ball_area_thresh  # threshold for a ball; should be less than pi/4
        self.bounds = bounds  # HSV bounds for detecting powercells
        # self.sync_post = sync_post  # post-processing to run in the same thread
        self.stream = None  # Video stream
        self.ball_radius = ball_radius  # radius of powercells in meters

        self.initialize_values()

        # --- initialized by .start() ---
        self.frame_width = None  # width of video stream
        self.frame_height = None  # height of video stream
        self.hfov = None  # horizontal FOV
        self.vfov = None  # Vertical FOV
        self.top_d = None  # distance the top of the frame covers on a flat ground
        self.bottom_d = None  # distance the bottom of the frame covers on a flat ground
        self.d_diff = None  # difference in the distance of the top vs. bottom of the frame
        self.top_w = None  # width of ground at top of frame
        self.bottom_w = None  # width of ground at bottom of frame
        self.w_diff = None  # difference in the width of the top vs. bottom of the frame

    def initialize_values(self):
        if self.preprocess_fn is None:
            self.preprocess_fn = self.DEFAULT_PREPROCESS
        else:
            self.preprocess_fn = self.preprocess_fn

        if self.bounds is None:
            self.bounds = self.DEFAULT_BOUNDS

    def get_balls(self, hsv_img: np.ndarray) -> np.ndarray:
        """
        Given an HSV image it can find the powercells. For best results a Gaussian blur on the image
        is recommended beforehand. This method identifies the x and y of the center of a ball
        (on the frame) and the radius of the ball, all in px.

        :param hsv_img: 3 channel HSV image
        :type hsv_img: np.ndarray
        :return: a list of the x, y, and radius of the balls, sorted top left to bottom right
        :rtype: np.ndarray
        """
        masks = map(lambda bound: cv2.inRange(hsv_img, bound[:3], bound[3:]), self.bounds)
        mask = reduce(cv2.bitwise_or, masks)  # applies all the masks from each bound

        # erodes the dilates the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, iterations=4, kernel=None)

        # finds contours in the resulting mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        hulls = list(map(cv2.convexHull, cnts))  # creates then draws all the convex hulls
        mask = cv2.drawContours(np.zeros(mask.shape, dtype='uint8'), hulls, -1, 255, -1)

        balls = []
        for i, contour in enumerate(cnts):
            (x, y), radius = cv2.minEnclosingCircle(contour)  # tentative values for the ball

            # isolating the contour
            bbox = np.array([[max(0, int(x - radius)), max(0, int(y - radius))], [min(len(mask[0]), int(x + radius) + 1), min(len(mask), int(y + radius) + 1)]])
            roi = mask[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
            # TODO: prevent two (or more) contours from being in the same ROI

            # keeps everything in the ROI that is within the circle
            circle_mask = cv2.circle(np.zeros(roi.shape), (roi.shape[1] // 2, roi.shape[0] // 2), int(radius), 1, -1)
            circle_mask = np.logical_and(circle_mask, roi).astype(float)

            # if there is a certain thresh of the contour that fits within a circle, it must be a
            # ball. Note: the maximum for this threshold is pi/4, i.e. the ratio between the area of
            # a circle radius r and the area of a (bounding) square, side length 2r
            if circle_mask.mean() > self.ball_area_thresh:
                balls.append((x, y, radius))

        return self.sort_top_l_to_bottom_r(balls)

    @staticmethod
    def sort_top_l_to_bottom_r(anns: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Sorts the balls from the top left to the bottom right, going across.

        :param anns: a list of the balls x, y, radius in px
        :type anns: List[Tuple[int, int, int]]
        :return: 2D sorted array
        :rtype: np.ndarray
        """
        return np.array(sorted(anns, key=lambda xyr: xyr[0] + (xyr[1] * 1000)))

    def start(self):
        """
        Starts the VideoStream and initializes all of the values used for computing the real
        position of the powercells. This is done just for ease of computation, I don't know if it
        actually improves performance.
        """
        self.stream = VideoStream(src=self.src)  # gets VideoStream from src

        # Frame height and width
        self.frame_width = self.stream.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.stream.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Horizontal and vertical FOV based off diagonal FOV and aspect ratio
        self.hfov = self.frame_width * self.fov / sqrt(self.frame_width**2 + self.frame_height**2)
        self.vfov = self.frame_height * self.fov / sqrt(self.frame_width**2 + self.frame_height**2)

        # distances that the pixels on the very top and bottom of the frame represent on flat ground
        self.top_d = -self.height/tan(radians(self.offset_angle))
        self.bottom_d = -self.height/tan(radians(self.offset_angle - self.vfov))
        self.d_diff = self.top_d - self.bottom_d  # difference in distances

        # width that the pixels on the very top and bottom of the frame represent on flat ground
        self.top_w = sqrt(self.height**2 + self.top_d**2) * sin(radians(self.hfov / 2)) / sin(radians(90 - self.hfov / 2))
        self.bottom_w = sqrt(self.height**2 + self.bottom_d**2) * sin(radians(self.hfov / 2)) / sin(radians(90 - self.hfov / 2))
        self.w_diff = self.top_w - self.bottom_w  # difference in widths

        self.stream.start()  # starts the VideoStream

    def stop(self):
        """
        Stops the video stream
        """
        self.stream.stop()

    def run(self, out_queue: Union[Queue, LifoQueue] = None, should_run: Callable[[int], bool] = lambda x: True):
        """
        Runs in a loop until `should_run()` returns False. Puts the ball data in `out_queue`, it is
        recommend to use a LIFO queue for this.

        :param out_queue: output queue
        :type out_queue: Union[Queue, LifoQueue]
        :param should_run: function to decide end of while loop, param is the amount of loops
        :type should_run: Callable[[int], bool]
        """
        self.start()
        i = 0
        while should_run(i):
            self.next_frame(out_queue=out_queue)
        self.stop()

    def next_frame(self, out_queue: Union[Queue, LifoQueue] = None) -> np.ndarray:
        """
        Runs the ball finder and then processes the values and gets the real position of balls from
        the next frame in the stream.

        :param out_queue: output queue
        :type out_queue: Union[Queue, LifoQueue]
        :return: the ball data
        :rtype: np.ndarray
        """

        frame = self.stream.read()  # gets next frame
        img = self.preprocess_fn(frame)  # preprocess (BGR -> HSV and maybe some blurring)
        balls = self.get_balls(img)  # looks for the balls

        ball_data = self.analyze_balls(balls)  # gets the real position of the balls

        if out_queue is not None:  # put in queue if queue exists
            out_queue.put((ball_data, frame, datetime.datetime.now()))

        return balls

    def analyze_balls(self, balls: np.ndarray) -> List[Dict[str, Dict[str, Union[int, float]]]]:
        """
        Gets the real posision of the balls based on their position in the frame and they camera's
        orientation.

        :param balls: array of the balls x, y, and radius in px
        :type balls: np.ndarray
        :return: the frame and real positions of the balls
        :rtype: List[Dict[str, Dict[str, Union[int, float]]]]
        """
        ball_data = []
        for x, y, r in balls:
            # distance the ball is from the bottom of the frame in %
            height_radio = (1 - y / self.frame_height)

            # the real x and y position of the ball
            # Note: the camera is at (0, 0) and it is facing positive x, so the farther somthing is,
            # the higher its x value. Any anything on the right will have a positive y while
            # anything on the left will have negative y
            global_x = self.height * tan(radians(self.vfov * height_radio + (90 + self.offset_angle - self.vfov)))
            global_y = (height_radio * self.w_diff + self.bottom_w) * ((x * 2/self.frame_width) - 1)

            # the real theta and radius position, this is probably more useful than x and y
            # Note: the camera is at (0, 0) and it is facing 0 deg, so anything to the right of the
            # camera has a positive angle, and anything on the left has a negative one
            global_r = sqrt(global_x**2 + global_y**2)
            global_theta = degrees(atan(global_y / global_x))

            ball_data.append({
                "px": {
                    "x": int(x),
                    "y": int(y),
                    "radius": int(r)
                },
                "pos": {
                    "x": global_x,
                    "y": global_y,
                    "radius": global_r,
                    "theta": global_theta
                }
            })
        return ball_data


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Default preprocessor, it blurs then converts the frame to HSV (from BGR).

    :param frame: image or frame from video
    :type frame: np.ndarray
    :return: processed image
    :rtype: np.ndarray
    """
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    return hsv


class LifoQueueManager(BaseManager):
    """
    If you want to use a LIFO Queue as the out Queue, you must get the queue from this manager.

    > manager = LifoQueueManager()
    > manager.start()
    > lifo_queue = manager.LifoQueue()
    """
    pass


LifoQueueManager.register('LifoQueue', LifoQueue)
