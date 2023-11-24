from cv2 import putText, FONT_HERSHEY_SIMPLEX, LINE_AA
from config import PHONE, CAMERA, SCREEN
from project import camera, gesturesNew
import threading
from utils import CvFpsCalc
import time
from turbojpeg import TurboJPEG
from config import TARGET_FPS

class Start:
    __slots__ = (
        "camera",
        "screen",
        "merge",
        "gesture",
        "cvFpsCalc",
        "fps",
        "timer",
    )

    def __init__(self):
        print("initializing")
        self.camera = camera.Camera(CAMERA, False)
        self.screen = camera.Camera(SCREEN, True)
        print("got cameras")
        self.gesture = gesturesNew.Gestures()
        print("got gestures")
        print("finished init")
        self.cvFpsCalc = CvFpsCalc(buffer_len=64)
        self.timer = 0

    def run(self):
        camera = self.camera.run()
        screen = self.screen.run()
        screen = self.gesture.run(camera, screen)
        fps = self.cvFpsCalc.get()
        putText(
            screen,
            "FPS:" + str(fps),
            (10, 30),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            (255,255,255),
            4,
            LINE_AA,
        )
        return screen


print("starting")
starter = Start()
print("started")

outputFrame = None
lock = threading.Lock()
result = None
jpeg = TurboJPEG('./libjpeg-turbo-gcc64/bin/libturbojpeg.dll')

def get_frames():
    global outputFrame, lock, result
    print("starting")
    delta = 1 / TARGET_FPS
    last_time = time.time()
    new_time = time.time()
    while True:
        new_time = time.time()
        if new_time - last_time < delta:
            continue
        last_time += delta
        frame = starter.run()
        with lock:
            outputFrame = frame


def generate():
    global outputFrame, lock, result
    print("generating")
    delta = 1 / TARGET_FPS
    while True:
        # compress image into buffer
        encodedImage = jpeg.encode(outputFrame, quality=75)
        result = bytearray()
        result.extend(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        result.extend(bytearray(encodedImage))
        result.extend(b"\r\n")
        time.sleep(delta)
        yield (bytes(result))
