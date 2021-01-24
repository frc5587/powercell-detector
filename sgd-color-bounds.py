import os
from functools import reduce
from pathlib import Path
import json
import random as r
import datetime

import numpy as np
import cv2


class SGDColorBoundsLearner:
    ball_area_thresh = 0.6

    def __init__(self, data, lr=10, lr_decay=0.8, max_value=256, nbounds=4, bs=None):
        self.lr = lr
        self.lr_decay = lr_decay
        self.data = data
        self.bs = len(data) if bs is None else bs

        self.epoch = 0
        self.best_loss = 10000
        self.best_bounds = None
        self.best_epoch = None

        # self.bounds = np.random.randint(0, max_value, (3, nbounds, 2))
        self.bounds = np.array([((0, 90, 197), (86, 255, 255)), ((0, 186, 164), (58, 255, 208)), ((5, 9, 243), (33, 255, 255)), ((5, 165, 35), (55, 255, 164))])[:nbounds]

    def predict(self, hsv_img):
        masks = map(lambda bound: cv2.inRange(hsv_img, bound[0], bound[1]), self.bounds)
        mask = reduce(cv2.bitwise_or, masks)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        balls = []
        for i, contour in enumerate(cnts):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            bbox = np.array([[max(0, int(x - radius)), max(0, int(y - radius))], [min(mask.shape[1], int(x + radius) + 1), min(mask.shape[0], int(y + radius) + 1)]])
            roi = mask[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

            circle_mask = cv2.circle(np.zeros(roi.shape), (roi.shape[1] // 2, roi.shape[1] // 2), int(radius), 1, -1)
            circle_mask = np.logical_and(circle_mask, roi).astype(float)
            if circle_mask.mean() > self.ball_area_thresh:
                balls.append((x, y, radius))

        return sort_top_l_to_bottom_r(balls)

    def get_loss(self):
        losses = []
        for inp, out in self.data:
            loss = 0
            preds = self.predict(inp)
            loss += 50 * abs(len(preds) - len(out))

            if len(preds) > 0 and len(out) > 0:
                preds, out = self.make_same_length(preds, out)
                loss += (((out - preds) ** 2).sum(1) ** 0.5).sum()
            losses.append(loss)

        return sum(losses) / len(losses)

    @staticmethod
    def make_same_length(one, two):
        if one.shape[0] > two.shape[0]:
            return one[:two.shape[0]], two
        elif one.shape[0] < two.shape[0]:
            return one, two[:one.shape[0]]
        else:
            return one, two

    def get_ball_acc(self):
        wrong = 0
        total = 0
        for inp, out in self.data:
            preds = self.predict(inp)
            wrong += abs(len(preds) - len(out))
            total += len(out)

        return (1 - wrong/total) * 100

    def fit(self, epochs):
        print("%5s | %-10s | %-10s | %-10s | %-10s" % ("epoch", "loss", "lr", "bacc", "time"))
        print("------------------------------------------------------")

        for e in range(epochs):
            then = datetime.datetime.now()
            old_loss = self.get_loss()

            for i, n in enumerate(self.bounds):
                for j, mm in enumerate(n):
                    for k, value in enumerate(mm):
                        self.bounds[i, j, k] += self.lr
                        self.check_bound(i, j, k)
                        new_loss = self.get_loss()
                        if new_loss > old_loss:
                            self.bounds[i, j, k] = value - self.lr
                            self.check_bound(i, j, k)
                            newer_loss = self.get_loss()
                            if newer_loss > old_loss:
                                self.bounds[i, j, k] = value

                        old_loss = new_loss

            if new_loss < self.best_loss:
                self.best_loss = new_loss
                self.best_bounds = self.get_bounds()
                self.best_epoch = e

            sec_diff = (datetime.datetime.now() - then).total_seconds()
            print("%5d | %-10.5f | %-10.5f | %-10.5f | %-10.5f" % (e, new_loss, self.lr, self.get_ball_acc(),sec_diff))

            self.lr *= self.lr_decay
            self.lr = max(1, self.lr)

    def get_bounds(self):
        return [tuple(bound[0] + bound[1]) for bound in self.bounds.tolist()]

    def check_bound(self, i, j, k):
        self.bounds[i, j, k] = max(0, min(self.bounds[i, j, k], 255))


def get_circle(bbox):
    xy = sum(bbox[:, 0])/2, sum(bbox[:, 1])/2
    radius = sum((bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])) / 4
    return (*xy, radius)


def sort_top_l_to_bottom_r(anns):
    return np.array(sorted(anns, key=lambda xyr: xyr[0] + (xyr[1]*1000)))


def get_data(img_path, ann_path):
    files = json.load(open(ann_path/'..'/'sample.json'))
    files = files['balls'] + files['ballsnt']
    data = []
    for img_file in files:
        ann = json.load(open(ann_path/(img_file + ".json")))
        img = cv2.imread(str(img_path/img_file), cv2.IMREAD_COLOR)
        # img = im.resize(img, width=600)
        img = cv2.GaussianBlur(img, (15, 15), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if len(ann['objects']) > 0:
            ann = [get_circle(np.array(obj["points"]["exterior"])) for obj in ann['objects']]
            ann = sort_top_l_to_bottom_r(ann)
        else:
            ann = np.array([])

        data.append((
            img,
            ann
        ))
    r.shuffle(data)
    return data


if __name__ == "__main__":
    image_path = Path("./powercell-data/img")
    annotation_path = Path("./powercell-data/ann")

    data_set = get_data(image_path, annotation_path)
    learner = SGDColorBoundsLearner(data_set, lr=5, lr_decay=0.9, nbounds=3)

    learner.fit(20)

    print(f"\n\nBest Bounds:\n{learner.best_bounds}\nFrom Epoch: {learner.best_epoch}")


