from config import (
    WIDTH,
    HEIGHT,
    IMAGE_PATH,
    IMAGE_PATH2,
    IMAGE_PATH3,
    IMAGE_PATH4
)
import datetime
from numpy import where, array
from cv2 import (
    waitKey,
    resizeWindow,
    resize,
    namedWindow,
    inRange,
    imread,
    getTrackbarPos,
    cvtColor,
    createTrackbar,
    COLOR_GRAY2BGR,
    COLOR_BGR2HSV,
    bitwise_and,
)


class GreenScreen:
    def __init__(self):
        self.img1 = resize(imread(IMAGE_PATH),(WIDTH,HEIGHT))
        self.img2 = resize(imread(IMAGE_PATH2),(WIDTH,HEIGHT))
        self.img3 = resize(imread(IMAGE_PATH3),(WIDTH,HEIGHT))
        self.img4 = resize(imread(IMAGE_PATH4),(WIDTH,HEIGHT))
        self.state = 0

    def nothing(self, one):
        pass

    def cvt_to_hsv(self):
        return cvtColor(self.frame,COLOR_BGR2HSV)
    
    def find_range(self):
        return inRange(self.hsv_frame, self.lower_range_green, self.upper_range_green)
    
    def bitwise(self):
        return bitwise_and(self.frame, self.frame ,mask=self.mask)
    
    def cvt_to_bgr(self):
        return cvtColor(self.mask,COLOR_GRAY2BGR)
    
    def find_where_pixels_is_equal_to_mask(self):
        self.timer = datetime.datetime.now().time().second
        if self.timer <= 15:
            return where(self.mask,self.img1,self.difference)
        elif self.timer >= 15 and self.timer <= 30:
            return where(self.mask,self.img2,self.difference)
        elif self.timer >= 30 and self.timer <= 45:
            return where(self.mask,self.img3,self.difference)
        elif self.timer >= 45 and self.timer <= 60:
            return where(self.mask,self.img4,self.difference)
    
    def trackbar(self):
        namedWindow("Trackbars")
        resizeWindow("Trackbars", 1000, 1000)
        createTrackbar("L-H", "Trackbars", 45, 179, self.nothing)
        createTrackbar("L-S", "Trackbars", 25, 255, self.nothing)
        createTrackbar("L-V", "Trackbars", 61, 255, self.nothing)
        createTrackbar("U-H", "Trackbars", 91, 179, self.nothing)
        createTrackbar("U-S", "Trackbars", 255, 255, self.nothing)
        createTrackbar("U-V", "Trackbars", 255, 255, self.nothing)
        self.state = 1

    def values(self):
        self.l_h = getTrackbarPos("L-H", "Trackbars")
        self.l_s = getTrackbarPos("L-S", "Trackbars")
        self.l_v = getTrackbarPos("L-V", "Trackbars")
        self.u_h = getTrackbarPos("U-H", "Trackbars")
        self.u_s = getTrackbarPos("U-S", "Trackbars")
        self.u_v = getTrackbarPos("U-V", "Trackbars")
        return self.l_h,self.l_s,self.l_v,self.u_h,self.u_s,self.u_v

    def run(self, frame):
        self.frame = frame
        if not self.state:
            self.trackbar()
        self.l_h,self.l_s,self.l_v,self.u_h,self.u_s,self.u_v = self.values()
        self.lower_range_green = array([self.l_h, self.l_s, self.l_v])
        self.upper_range_green = array([self.u_h, self.u_s, self.u_v])
        self.hsv_frame = self.cvt_to_hsv()
        self.mask = self.find_range()
        self.difference = self.frame - self.bitwise()
        self.mask = self.cvt_to_bgr()
        self.frame = self.find_where_pixels_is_equal_to_mask()
        if waitKey(1) & 0xFF == ord('q'):
            pass
        return self.frame