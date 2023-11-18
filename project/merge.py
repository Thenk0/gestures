import cv2
import cupy as np
import pyautogui
import time

SCREEN_SIZE = (1920, 1080)

fps = 30
prev = 0

while True:
    
    time_elapsed = time.time()-prev
    img = pyautogui.screenshot()
    if time_elapsed > 1.0/fps:
        prev = time.time()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    cv2.waitKey(0)