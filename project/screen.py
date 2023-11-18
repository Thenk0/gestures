from config import WIDTH, HEIGHT, SCREEN_WIDTH,SCREEN_HEIGHT, TARGET_FPS
import dxcam
from numpy import asarray
import numpy as np
from cv2 import resize
from PIL import Image
import cv2, threading
import queue
import time
from .screenCapture import ScreenCapture

# bufferless VideoCapture
class Screen:
    def __init__(self):
        self.bounding = {
            "top": 0,
            "left": 0,
            "width": WIDTH,  # 1920
            "height": HEIGHT,  # 1080
        }
        # self.camera = dxcam.create(output_idx=0, output_color="BGRA")
        # self.camera.start(target_fps=TARGET_FPS, video_mode=True)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()


    def convert_screen_frame_to_array(self):
        return asarray(self.frame)

    def resize(self, frame):
        return resize(frame, (WIDTH, HEIGHT))

    def run(self, capture):
        self.frame = capture.screenshot()
        if SCREEN_WIDTH != WIDTH:
            self.frame = self.resize()
        return self.frame

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        with ScreenCapture((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)) as capture:
            while True:
                frame = capture.screenshot()
                if SCREEN_WIDTH != WIDTH:
                    frame = self.resize(frame)  
                if not self.q.empty():
                    try:
                        self.q.get_nowait()  # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                frame = frame.copy()
                self.q.put(frame)
                time.sleep(1/TARGET_FPS)

    def read(self):
        return self.q.get()
