from copy import deepcopy
from cv2 import imencode
from Project import camera, gesture, greenscreen, merge, screen
import threading

class Start():
    __slots__ = ["camera","screen","merge","gesture","greenscreen"]

    def __init__(self):
        self.camera = camera.Camera()
        self.screen = screen.Screen()
        self.merge = merge.Merge()
        self.gesture = gesture.Gesture()
        self.greenscreen = greenscreen.GreenScreen()
    
    def run(self):
        while 1:
            camera = self.camera.run()
            camera = self.gesture.run(camera)
            greenscreen = self.greenscreen.run(camera)
            screen = self.screen.run()
            merge = self.merge.run(greenscreen,screen)
            return merge

starter = Start()
outputFrame = None
lock = threading.Lock()

def get_frames():
    global outputFrame,lock
    while 1:
        frame = starter.run()
        with lock:     
            outputFrame = deepcopy(frame)

def generate():
    global outputFrame, lock
    while 1:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
        	bytearray(encodedImage) + b'\r\n')