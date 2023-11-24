WIDTH, HEIGHT = 1920, 1080
VERTICAL_FLIP = 1 
"""See opencv2 flip https://www.geeksforgeeks.org/python-opencv-cv2-flip-method/"""
DO_FLIP = True

# Ids of cameras, note that screen is a virtual OBS camera
CAMERA = 1
SCREEN = 2
PHONE = "http://192.168.31.107:8080/video" # IP camera address

FLIP_HANDS = True
"""Flip hand labels if camera sees hands forward"""
GESTURE_ACCURACY = 5
"""Amount of frames gesture needs to be repeated to take action"""

SCROLL_DIST_DEAD_ZONE = 10
"""Amount of pixel difference for scroll to take effect"""
SCROLL_MULTIPLIER = 20
"""How much pixels for a tick of scroll"""

CURSOR_DIST_DEAD_ZONE = 5
"""Amount of pixel difference from previous mouse position to move"""
CURSOR_GRAB = 20
"""How much frames should pass before cursor automatically clicks"""

TARGET_FPS = 30