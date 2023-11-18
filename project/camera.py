from config import (
    WIDTH,
    HEIGHT,
    VERTICAL_FLIP,
)
from cv2 import (
    flip,
    resize,
)
from .bufferlessCapture import BufferlessVideoCapture
from numpy import asarray

class Camera:

    __slots__ = ("cap", "frame", "is_screen")
    
    def __init__(self, camera, screen):
        self.is_screen = screen
        self.cap = BufferlessVideoCapture(camera, screen) # PHONE, CAMERA

    def capture_camera(self):
        return self.cap.read()
    
    def flip(self):
        return flip(self.frame,VERTICAL_FLIP)
        
    def resize(self):
        return resize(self.frame, (WIDTH, HEIGHT))
    
    def convert_to_array(self):
        return asarray(self.frame)
    
    def run(self):
        self.frame = self.capture_camera()
        if not self.is_screen:
            self.frame = self.flip()
        self.frame = self.resize()
        self.frame = self.convert_to_array()
        return self.frame