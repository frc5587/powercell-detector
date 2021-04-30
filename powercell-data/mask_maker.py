import os
import json

import numpy as np
import cv2


def make_mask_folder(path="./msk"):
    if os.path.exists(path):
        return False
    else:
        os.mkdir(path)
        return True


def make_mask(img_name):
    ann_name = img_name + ".json"
    msk_name = "msk_" + img_name

    ann = json.load(open("./ann/" + ann_name))
    img = cv2.imread("./img/" + img_name, cv2.IMREAD_COLOR)
    msk = np.zeros(img.shape[:2])
    print(msk.shape)

    for obj in ann["objects"]:
        points = np.array(obj['points']['exterior'])
        center = (points[0] + (points[1] - points[0])/2).astype(int)
        r = int(((points[1] - points[0])/2).mean())

        msk = cv2.circle(msk, tuple(center), r, 1, -1, )

    cv2.imwrite("./msk/" + msk_name, msk)


def main():
    make_mask_folder()
    img_files = os.listdir("./img")

    for img_file in img_files:
        make_mask(img_file)


if __name__ == '__main__':
    main()
