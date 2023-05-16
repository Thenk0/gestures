from cv2 import (
    cvtColor,
    COLOR_RGB2BGR,
    COLOR_BGR2RGB,
    addWeighted,
)
from numpy import asarray

class Merge:
    __slots__ = ["frame","hands", "news", "imgs", "imgs1", "news1", "news2"]

    def convert_to_array(self) -> tuple:
        return asarray(self.news), asarray(self.imgs)
    
    def convert_color(self):
        return cvtColor(self.news, COLOR_RGB2BGR)
    
    def convert_color2(self):
        return cvtColor(self.news, COLOR_BGR2RGB)

    def merge(self): 
        return addWeighted(self.news, 0.5, self.imgs, 1, 0)
    
    def run(self, imgs, news) -> tuple:
        self.imgs = imgs
        self.news = news
        self.news, self.imgs = self.convert_to_array()
        self.news = self.convert_color()
        self.news = self.convert_color2()
        self.frame = self.merge()
        return self.frame