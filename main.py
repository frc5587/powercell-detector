from functools import reduce
from pathlib import Path
import numpy as np
import os
import json
import datetime

import cv2
import imutils as im
from imutils.video import VideoStream


img_path = Path("./powercell-data/img")
ann_path = Path("./powercell-data/ann")
use_cam = False
manual = True

# hsv_bounds = [(24, 91, 199, 123, 255, 255), (18, 172, 182, 91, 255, 245), (26, 30, 233, 70, 255, 255), (22, 159, 57, 79, 255, 182)]  # v3
hsv_bounds = [(22, 119, 190, 152, 255, 255), (54, 222, 197, 100, 255, 255), (27, 27, 218, 99, 255, 255), (15, 156, 83, 121, 255, 230)]  # v4
ball_area_thresh = .6


def get_ball(img, show=False):
    frame = im.resize(img, width=600)
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    masks = [cv2.inRange(hsv, bound[:3], bound[3:]) for bound in hsv_bounds]
    mask = reduce(cv2.bitwise_or, masks)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = im.grab_contours(cnts)

    balls = []
    for i, contour in enumerate(cnts):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        bbox = np.array([[max(0, int(x - radius)), max(0, int(y-radius))], [min(len(mask[0]), int(x + radius) + 1), min(len(mask), int(y + radius) + 1)]])
        roi = mask[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

        circle_mask = np.zeros(roi.shape)

        circle_mask = cv2.circle(circle_mask, (len(circle_mask[0]) // 2, len(circle_mask) // 2), int(radius), 1, -1)
        circle_mask = np.logical_and(circle_mask, roi).astype(float)

        if circle_mask.mean() > ball_area_thresh:
            balls.append((x, y, radius))
            if show:
                frame = cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    if show:
        mask = np.stack((mask for _ in range(3)), 2)
        cv2.imshow("Stream", np.hstack([frame, mask]))
        return balls
    else:
        return balls


def main():
    accuracy = []
    wrong_balls = 0
    total_balls = 0

    if use_cam:
        vs = VideoStream(src=0).start()
        then = datetime.datetime.now()
        frames = 0
        while True:
            frames += 1
            fr = vs.read()
            get_ball(fr, show=True)
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
        vs.stop()
    else:
        then = datetime.datetime.now()
        for frames, file in enumerate(os.listdir(img_path), start=1):
            ann = json.load(open(ann_path/(file + ".json"), 'r'))
            gt_balls = len(ann["objects"])
            fr = cv2.imread(str(img_path/file))
            balls = get_ball(fr, show=manual)

            wrong = abs(gt_balls - len(balls))
            wrong_balls += wrong
            total_balls += gt_balls

            if wrong == 0:
                accuracy.append(1)
            elif wrong > 0 and gt_balls == 0:
                accuracy.append(0)
            else:
                accuracy.append(1-(wrong/gt_balls))

            if manual:
                print(file)
                cv2.waitKey(0)

    secs = (datetime.datetime.now() - then).seconds

    print("time:", secs)
    print("pics/sec:", frames / secs)
    print("accuracy:", "n/a" if len(accuracy) == 0 else np.array(accuracy).mean())
    print("ball accuracy:", "n/a" if total_balls == 0 else (1 - wrong_balls/total_balls))


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
