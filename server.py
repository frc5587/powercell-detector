import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from multiprocessing import Process, Event
from queue import Empty
import datetime
from time import sleep

import numpy as np
from flask import Flask, Response, render_template
from math import tan, radians
import cv2
from networktables import NetworkTablesInstance, NetworkTables

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


def on_connected(*args):
    print("CONNECTED", args)
    connected_evt.set()


def send_data(vqueue):
    NetworkTables.initialize(server='10.55.87.2')
    NetworkTables.addConnectionListener(on_connected)
    
    connected_evt.wait()
    
    table = NetworkTables.getTable("powercellDetector")
    
    test = NetworkTables.getTable("SmartDashboard")
    print(test.getEntry("gyro").getDouble(0))
    
    tn = table.getEntry("tn")
    tr = table.getEntry("tr")
    tx = table.getEntry("tx")
    ty = table.getEntry("ty")
    ttheta = table.getEntry("ttheta")

    last_time = datetime.datetime.now()

    while True:
        ball_data, frame, time = vqueue.get()
        ball_data: list
        if time < last_time:  # check if frame is newer than the most recent one
            continue
        else:
            last_time = time

        if not ball_data:
            tn.setDouble(0)
            tr.setDouble(0)
            tx.setDouble(0)
            ty.setDouble(0)
            ttheta.setDouble(0)

        else:
            ball_data.sort(key=lambda x: x['pos']['radius'])
            closest_ball = ball_data[0]

            tn.setDouble(len(ball_data))
            tr.setDouble(closest_ball['pos']['radius'])
            tx.setDouble(closest_ball['pos']['x'])
            ty.setDouble(closest_ball['pos']['y'])
            ttheta.setDouble(closest_ball['pos']['theta'])

def test_cam(src):
    cap = cv2.VideoCapture(src)
    if cap is None:
        return False
    elif not cap.isOpened():
        cap.release()
        return False
    else:
        cap.release()
        return True

if __name__ == '__main__':
    connected_evt = Event()
    while not test_cam(0):
        sleep(1)
        print("Waiting for camera connection...")
    
    manager = LifoQueueManager()
    manager.start()
    video_q = manager.LifoQueue(maxsize=10)
    
    pc_finder = PowercellFinder(0, preprocess_fn=preprocess, height=.65, offset_angle=-8.5)

    pc_finder_proc = Process(target=pc_finder.run, kwargs=dict(out_queue=video_q))
    pc_finder_proc.start()

    data_sender_proc = Process(target=send_data, args=(video_q,))
    data_sender_proc.start()

    app.run(host="0.0.0.0", port=80)
