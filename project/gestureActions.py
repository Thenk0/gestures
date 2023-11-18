from config import WIDTH, HEIGHT
import math
import pyautogui
import win32gui
# for why is it 9, refer to media_pipe index description
HAND_CENTER_POINT = 9
HAND_THUMB_END = 4

SCROLL_DIST_DEAD_ZONE = 10
SCROLL_MULTIPLIER = 20

CURSOR_DIST_DEAD_ZONE = 5
CURSOR_TYPES = {
    "auto": 0,
    "pointer": 0,
    "grab": 0,
    "move": 0
}

can_click = True

previous_scroll_dist = [0, 0]
held_buttons = []



pyautogui.PAUSE = 0

def get_cursor_type():
    return win32gui.GetCursorInfo()[1]

def reset_buttons():
    global held_buttons
    for button in held_buttons:
        pyautogui.keyUp(button)
        
    held_buttons = []

def resets_buttons(func):
    def wrapper():
        reset_buttons()
        func()
    return wrapper

class GestureActions:
    """
    All returns from these class methods are a bool
    @returns
    True repeating action
    False non repeating action
    """

    def __init__(self):
        """
        left [0], right [1]
        """
    @staticmethod
    @resets_buttons
    def scroll(gestures):
        global previous_scroll_dist
        left_gesture = gestures[0] if gestures[0]["label"] == "Left" else gestures[1]
        right_gesture = gestures[0] if gestures[0]["label"] == "Right" else gestures[1]

        left_x, left_y, left_z = left_gesture["landmarks"][HAND_THUMB_END]
        right_x, right_y, right_z = right_gesture["landmarks"][HAND_THUMB_END]
        del left_z, right_z

        screen_center = (int(WIDTH / 2), int(HEIGHT / 2))
        screen_center_x, screen_center_y = screen_center

        # basically we want left hand to be on the left, and right hand to be on the right
        # so we can prevent some strange scroll behavior
        if left_x > screen_center_x or right_x < screen_center_x:
            return True

        left_dist = math.hypot(screen_center_x - left_x, screen_center_y - left_y)
        right_dist = math.hypot(screen_center_x - right_x, screen_center_y - right_y)

        left_previous_scroll_dist = previous_scroll_dist[0]
        right_previous_scroll_dist = previous_scroll_dist[1]

        overall_dist = left_dist + right_dist
        prev_dist = left_previous_scroll_dist + right_previous_scroll_dist

        # here we limit scrolling
        dist_diff = abs(overall_dist) - abs(prev_dist)
        if -SCROLL_DIST_DEAD_ZONE < dist_diff < SCROLL_DIST_DEAD_ZONE:
            return True

        # scroll n times
        multi = int(abs(dist_diff) // SCROLL_MULTIPLIER)
        multi = max(1, min(multi, 5))
        if overall_dist > prev_dist:
            [pyautogui.scroll(100) for _ in range(0, multi)]
        if overall_dist < prev_dist:
            [pyautogui.scroll(-100) for _ in range(0, multi)]
        previous_scroll_dist = [left_dist, right_dist]
        return True

    @staticmethod
    @resets_buttons
    def cursor(gesture):
        position_x, position_y, _ = gesture["landmarks"][HAND_CENTER_POINT]
        del _

        position_x = (position_x / WIDTH) * 1920
        position_y = (position_y / HEIGHT) * 1080
        cursor_position_x, cursor_position_y = pyautogui.position()

        difference = (position_x - cursor_position_x) + (position_y - cursor_position_y)

        # limit cursor movement to min of pixels to limit jitter
        if -CURSOR_DIST_DEAD_ZONE < difference < CURSOR_DIST_DEAD_ZONE:
            return True

        pyautogui.moveTo(position_x, position_y)
        return True

    @staticmethod
    @resets_buttons
    def click(gesture):
        pyautogui.leftClick()
        return False

    @staticmethod
    @resets_buttons
    def right_click(gesture):
        pyautogui.rightClick()
        return False

    @staticmethod
    def _keyboard(button, gestures):
        global held_buttons
        # release all buttons except current
        for btn in held_buttons:
            if btn == button:
                continue
            pyautogui.keyUp(btn)
            held_buttons.remove(btn)
        

        if not button in held_buttons:
            pyautogui.keyDown(button)
            held_buttons.append(button)
        return True

    @staticmethod
    def left_first_group(gestures):
        return GestureActions._keyboard("E", gestures)

    @staticmethod
    def left_second_group(gestures):
        return GestureActions._keyboard("W", gestures)

    @staticmethod
    def left_third_group(gestures):
        return GestureActions._keyboard("Q", gestures)

    @staticmethod
    def right_first_group(gestures):
        return GestureActions._keyboard("R", gestures)

    @staticmethod
    def right_second_group(gestures):
        return GestureActions._keyboard("T", gestures)

    @staticmethod
    def right_third_group(gestures):
        return GestureActions._keyboard("Y", gestures)
    
    @staticmethod
    @resets_buttons
    def nothing():
        """
        yes i am really gonna leave this here, because it's funny \n
        gosha dembel lives in our code forever 
        """

    