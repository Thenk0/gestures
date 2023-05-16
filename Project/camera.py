from config import (
    WIDTH,
    HEIGHT,
    VERTICAL_FLIP,
    PHONE,
    CAMERA
)
from cv2 import (
    VideoCapture,
    flip,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
)

class Camera:
    __slots__ = ["cap","frame"]
    
    def __init__(self):
        self.cap = VideoCapture(CAMERA) # PHONE, CAMERA
        self.cap.set(CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(CAP_PROP_FRAME_HEIGHT,HEIGHT)

    def capture_camera(self):
        _, frame = self.cap.read()
        return frame
    
    def flip(self):
        return flip(self.frame, VERTICAL_FLIP)
    
    def run(self):
        self.frame = self.capture_camera()
        self.frame = self.flip()
        return self.frame