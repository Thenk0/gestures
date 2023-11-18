# import datetime
from numpy import where, array,asarray
from config import (
    WIDTH,
    HEIGHT,
)
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
    # GaussianBlur,
    # filter2D,
)


class GreenScreen:
    __slots__ = ("state",
                "frame",
                "l_h",
                "l_s",
                "l_v",
                "u_h",
                "u_s",
                "u_v",
                "lower_range_green",
                "upper_range_green",
                "hsv_frame",
                "mask",
                "difference",
                "timer",
                )

    def __init__(self):
        self.state = 0

    def nothing(self, one):
        pass

    def cvt_frame_to_hsv(self):
        return cvtColor(self.frame, COLOR_BGR2HSV)
    
    def find_range_of_green_color(self):
        return inRange(self.hsv_frame, self.lower_range_green, self.upper_range_green)
    
    def bitwise(self):
        return bitwise_and(self.frame, self.frame , mask=self.mask)
    
    def cvt_mask_to_bgr(self):
        return cvtColor(self.mask, COLOR_GRAY2BGR)
    
    def find_where_pixels_is_equal_to_mask(self):
        return where(self.mask, self.img1, self.difference)
       
    def create_trackbar(self):
        namedWindow("Trackbars")
        resizeWindow("Trackbars", 300, 300)
        createTrackbar("L-H", "Trackbars", 45, 179, self.nothing)
        createTrackbar("L-S", "Trackbars", 25, 255, self.nothing)
        createTrackbar("L-V", "Trackbars", 61, 255, self.nothing)
        createTrackbar("U-H", "Trackbars", 91, 179, self.nothing)
        createTrackbar("U-S", "Trackbars", 255, 255, self.nothing)
        createTrackbar("U-V", "Trackbars", 255, 255, self.nothing)
        self.state = 1

    def get_values(self):
        self.l_h = getTrackbarPos("L-H", "Trackbars")
        self.l_s = getTrackbarPos("L-S", "Trackbars")
        self.l_v = getTrackbarPos("L-V", "Trackbars")
        self.u_h = getTrackbarPos("U-H", "Trackbars")
        self.u_s = getTrackbarPos("U-S", "Trackbars")
        self.u_v = getTrackbarPos("U-V", "Trackbars")
        return self.l_h,self.l_s,self.l_v,self.u_h,self.u_s,self.u_v

    def run(self, frame):
        self.frame = frame
        # self.frame = self.blur()
        # self.frame = self.filter()

        # if not self.state:
        #     self.create_trackbar()
        self.l_h, self.l_s, self.l_v, self.u_h, self.u_s, self.u_v = self.get_values()
        # self.lower_range_green = array([self.l_h, self.l_s, self.l_v])
        # self.upper_range_green = array([self.u_h, self.u_s, self.u_v])
        
        self.lower_range_green = asarray([45,25,61])
        self.upper_range_green = asarray([91,255,255])

        self.hsv_frame = resize(self.cvt_frame_to_hsv(), (640, 480))
        self.mask = resize(self.find_range_of_green_color(), (WIDTH, HEIGHT))
        self.difference = self.frame - self.bitwise()
        self.mask = self.cvt_mask_to_bgr()
        self.frame = self.find_where_pixels_is_equal_to_mask()
        if waitKey(1) & 0xFF == ord('q'):
            pass
        return self.frame