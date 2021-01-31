from multiprocessing import Process
from queue import Empty
import datetime

import numpy as np
from flask import Flask, Response, render_template
from math import tan, radians
import cv2

from finder import PowercellFinder, preprocess, LifoQueueManager

DEFAULT_IMG = np.zeros((720, 1280, 3))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stream')
def stream():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def generate():
    last_time = datetime.datetime.now()
    times = []
    while True:
        try:
            ball_data, frame, time = video_q.get(timeout=0.5)

            if time < last_time:  # check if frame is newer than the most recent one
                continue
            else:
                times.append((time - last_time).total_seconds())
                last_time = time

            avg = 1 if sum(times) == 0 else (sum(times)/len(times))

            frame = cv2.line(frame, (0, len(frame)//2), (len(frame[0]), len(frame)//2), (0, 0, 0), 2)
            frame = cv2.line(frame, (len(frame[0])//2, 0), (len(frame[0])//2, len(frame)), (0, 0, 0), 2)

            frame = cv2.putText(frame, "%2.ffps" % (1/avg), (0, len(frame)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, "%2.ffps" % (1/avg), (0, len(frame)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            for ball in ball_data:
                try:
                    angle_endpoint = int((ball['px']['x'] - len(frame[0])//2)/tan(radians(ball['pos']['theta']))) + ball['px']['y']
                except ZeroDivisionError:
                    angle_endpoint = len(frame)

                frame = cv2.line(frame, (ball['px']['x'], ball['px']['y']), (len(frame[0])//2, angle_endpoint), (0, 255, 0), 2)
                frame = cv2.circle(frame, (ball['px']['x'], ball['px']['y']), ball['px']['radius'], (0, 255, 0), 2)

                frame = cv2.putText(frame, "%4.1fdeg" % ball["pos"]["theta"], (ball['px']['x'], ball['px']['y']), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
                frame = cv2.putText(frame, "%4.1fdeg" % ball["pos"]["theta"], (ball['px']['x'], ball['px']['y']), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                frame = cv2.putText(frame, "%4.2fm" % ball["pos"]["radius"], (ball['px']['x'], ball['px']['y'] + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
                frame = cv2.putText(frame, "%4.2fm" % ball["pos"]["radius"], (ball['px']['x'], ball['px']['y'] + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        except Empty:
            frame = DEFAULT_IMG

        if len(times) == 10:
            del times[0]

        flag, img = cv2.imencode(".jpg", frame)

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n'


if __name__ == '__main__':
    manager = LifoQueueManager()
    manager.start()
    video_q = manager.LifoQueue()

    # pc_finder = PowercellFinder(0, preprocess_fn=preprocess)
    pc_finder = PowercellFinder(0, preprocess_fn=preprocess, height=.75, offset_angle=-12)  # testing

    proc = Process(target=pc_finder.run, kwargs=dict(out_queue=video_q))
    proc.start()

    app.run(host="0.0.0.0", port=80)
