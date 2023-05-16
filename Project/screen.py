from config import (
    WIDTH,
    HEIGHT,
)
from mss import mss
from numpy import asarray

class Screen:
    __slots__ = ["bounding","frame"]

    def __init__(self):
        self.bounding = {
            "top": 0,
            "left": 0,
            "width": WIDTH,
            "height": HEIGHT,
        }

    def capture_screen(self):
        return mss().grab(self.bounding)
    
    def convert_frame_to_array(self):
        return asarray(self.frame)
    
    def run(self):
        self.frame = self.capture_screen()
        self.frame = self.convert_frame_to_array()
        return self.frame