from functools import reduce, partial
from queue import LifoQueue
from multiprocessing.managers import BaseManager
import numpy as np
import datetime
from math import tan, radians, degrees, sqrt, sin, atan

from imutils.video import VideoStream
import cv2


class PowercellFinder:
    DEFAULT_BOUNDS = [(24, 121, 190, 78, 247, 247), (66, 234, 209, 81, 243, 243), (27, 27, 218, 79, 245, 255), (20, 156, 83, 60, 253, 220)]  # v6
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

    def get_balls(self, hsv_img):
        masks = map(lambda bound: cv2.inRange(hsv_img, bound[:3], bound[3:]), self.bounds)
        mask = reduce(cv2.bitwise_or, masks)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, iterations=4, kernel=None)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        hulls = list(map(cv2.convexHull, cnts))
        mask = cv2.drawContours(np.zeros(mask.shape, dtype='uint8'), hulls, -1, 255, -1)

        balls = []
        for i, contour in enumerate(cnts):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            bbox = np.array([[max(0, int(x - radius)), max(0, int(y - radius))], [min(len(mask[0]), int(x + radius) + 1), min(len(mask), int(y + radius) + 1)]])
            roi = mask[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

            circle_mask = cv2.circle(np.zeros(roi.shape), (roi.shape[1] // 2, roi.shape[0] // 2), int(radius), 1, -1)
            circle_mask = np.logical_and(circle_mask, roi).astype(float)
            if circle_mask.mean() > self.ball_area_thresh:
                balls.append((x, y, radius))

        return self.sort_top_l_to_bottom_r(balls)

    @staticmethod
    def postprocess(fn, q):
        while True:
            balls, img = q.get(True)
            fn(balls, img)

    @staticmethod
    def sort_top_l_to_bottom_r(anns):
        return np.array(sorted(anns, key=lambda xyr: xyr[0] + (xyr[1] * 1000)))

    def start(self):
        self.stream = VideoStream(src=self.src)
        self.frame_width = self.stream.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.stream.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.hfov = self.frame_width * self.fov / sqrt(self.frame_width**2 + self.frame_height**2)
        self.vfov = self.frame_height * self.fov / sqrt(self.frame_width**2 + self.frame_height**2)

        self.top_d = -self.height/tan(radians(self.offset_angle))
        self.bottom_d = -self.height/tan(radians(self.offset_angle - self.vfov))
        self.d_diff = self.top_d - self.bottom_d

        self.top_w = sqrt(self.height**2 + self.top_d**2) * sin(radians(self.hfov / 2)) / sin(radians(90 - self.hfov / 2))
        self.bottom_w = sqrt(self.height**2 + self.bottom_d**2) * sin(radians(self.hfov / 2)) / sin(radians(90 - self.hfov / 2))
        self.w_diff = self.top_w - self.bottom_w

        self.stream.start()

    def stop(self):
        self.stream.stop()

    def run(self, out_queue=None, should_run=lambda x: True):
        self.start()
        i = 0
        while should_run(i):
            self.next_frame(out_queue=out_queue)
        self.stop()

    def next_frame(self, out_queue=None):
        frame = self.stream.read()
        img = self.preprocess_fn(frame)
        balls = self.get_balls(img)

        ball_data = self.analyze_balls(balls)

        if out_queue is not None:
            out_queue.put((ball_data, frame, datetime.datetime.now()))

        return balls

    def analyze_balls(self, balls):
        ball_data = []
        for x, y, r in balls:
            height_radio = (1 - y / self.frame_height)
            global_y = height_radio * self.d_diff + self.bottom_d
            global_x = (height_radio * self.w_diff + self.bottom_w) * ((x * 2/self.frame_width) - 1)

            global_r = sqrt(global_x**2 + global_y**2)  # TODO: this is not correct maybe global_x & y are wrong??
            global_theta = degrees(atan(global_x / global_y))

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
                },
                "extras": {
                    "top d": self.top_d,
                    "bottom d": self.bottom_d
                }
            })
        return ball_data


def preprocess(frame):
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    return hsv


def postprocess(balls, img):
    pass


def sync_postprocess(balls, img):
    cv2.imshow("Stream", img)
    cv2.waitKey(1)


class MyManager(BaseManager):
    pass


MyManager.register('LifoQueue', LifoQueue)
