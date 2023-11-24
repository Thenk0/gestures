from config import WIDTH, HEIGHT, VERTICAL_FLIP, DO_FLIP
from cv2 import (
    flip,
    resize,
)
from .bufferlessCapture import BufferlessVideoCapture
from numpy import asarray


class Camera:
    __slots__ = ("cap", "is_screen")

    def __init__(self, camera, screen):
        self.is_screen = screen
        self.cap = BufferlessVideoCapture(camera, screen)

    def run(self):
        frame = self.cap.read()
        if not self.is_screen and DO_FLIP:
            frame = flip(frame, VERTICAL_FLIP)
        frame = resize(frame, (WIDTH, HEIGHT))
        frame = asarray(frame)
        return frame
