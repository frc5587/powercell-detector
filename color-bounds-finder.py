from functools import reduce, partial
import cv2
import argparse
from operator import xor
import numpy as np
import imutils as im

ball_area_thresh = 0.6

windows = []

# vals = [(25, 128, 188, 34, 255, 255), (0, 223, 165, 63, 255, 255)]
# vals = [(22, 119, 190, 152, 255, 255), (54, 222, 197, 100, 255, 255), (27, 27, 218, 99, 255, 255), (15, 156, 83, 121, 255, 230)]
# vals = [(22, 119, 190, 152, 255, 255), (54, 222, 197, 100, 255, 255), (27, 27, 218, 99, 255, 255), (15, 156, 83, 72, 255, 230)]

DEFAULT_VAL = (0, 0, 0, 255, 255, 255)
vals = [DEFAULT_VAL]


def update_thresh(value):
    global ball_area_thresh
    ball_area_thresh = value / 100


def make_thresh_trackbar():
    cv2.namedWindow("Thresh")
    cv2.createTrackbar("Threshold", "Thresh", int(ball_area_thresh*100), 100, update_thresh)


def find_balls(frame, mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(filter(lambda x: cv2.contourArea(x) > 50, im.grab_contours(cnts)))

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
            frame = cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    return frame, cnts


def in_range(frame, hsv_values):
    return cv2.inRange(frame, hsv_values[:3], hsv_values[3:])


def callback(value):
    pass


def btn_callback(range_filter, value):
    for i in range(len(windows), max(value, len(windows))):
        setup_trackbars(range_filter, i + 1, DEFAULT_VAL)


def setup_trackbars(range_filter, num, val):
    global windows

    name = "Trackbars" + str(num)
    cv2.namedWindow(name, 0)
    windows.append(name)

    for i, mm in enumerate(["MIN", "MAX"]):
        # v = 0 if i == "MIN" else 255

        for j, rg in enumerate(range_filter):
            v = val[i * 3 + j]
            cv2.createTrackbar("%s_%s" % (rg, mm), name, v, 255, callback)
    if num == 1:
        cv2.createTrackbar("ranges", name, 1, 10, partial(btn_callback, range_filter))


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False,
                    help='Path to the image')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image after applying the mask',
                    action='store_true')
    args = vars(ap.parse_args())

    if not xor(bool(args['image']), bool(args['webcam'])):
        ap.error("Please specify only one image source")

    return args


def get_trackbar_values(range_filter, window):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), window)
            values.append(v)

    return tuple(values)


def main():
    args = get_arguments()

    range_filter = "HSV"

    if args['image']:
        image = cv2.imread(args['image'])
    else:
        camera = cv2.VideoCapture(1)

    make_thresh_trackbar()

    for i, val in enumerate(vals, start=1):
        setup_trackbars(range_filter, i, val)

    last_values = None

    while True:
        if args['webcam']:
            ret, image = camera.read()

            if not ret:
                break

        trackbar_values = [get_trackbar_values(range_filter, w) for w in windows]
        # image_to_mask = cv2.bilateralFilter(image, 5, 175, 175)

        image_to_mask = cv2.GaussianBlur(image, (15, 15), 0)
        image_to_mask = cv2.cvtColor(image_to_mask, cv2.COLOR_BGR2HSV)

        if trackbar_values != last_values:
            print(trackbar_values)
            last_values = trackbar_values

        masks = [in_range(image_to_mask, t) for t in trackbar_values]

        mask = reduce(cv2.bitwise_or, masks)
        mask_c = mask.copy()

        if args['preview']:
            preview = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow("Preview", preview)
        else:
            preview = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow("Preview", preview)

            blurryballs, cnts = find_balls(cv2.cvtColor(image_to_mask.copy(), cv2.COLOR_HSV2BGR), mask)
            cv2.imshow("Blur", blurryballs)

            balls, cnts = find_balls(image.copy(), mask)
            cv2.imshow("Ball Finder", balls)

            contours = cv2.drawContours(balls, cnts, -1, (255, 0, 0))
            cv2.imshow("Contours", contours)

            cv2.imshow("Mask", mask)
            cv2.imshow("Mask no tfms", mask_c)

            if len(masks) > 1:
                for i, m in enumerate(masks, start=1):
                    cv2.imshow(f"Mask{i}", m)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break


if __name__ == '__main__':
    main()
