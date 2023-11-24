from config import WIDTH, HEIGHT, CURSOR_DIST_DEAD_ZONE, CURSOR_GRAB, SCROLL_DIST_DEAD_ZONE, SCROLL_MULTIPLIER
import math
import pyautogui
import pydirectinput
import win32gui

# for index numbers refer to media_pipe index description
HAND_CENTER_POINT = 9
HAND_THUMB_END = 4

CURSOR_TYPES = {
    "auto": 65539,
    "pointer": 65567,
    "move": 65557,
}

cursor_grab_counter = 0
previous_scroll_dist = [0, 0]
previous_menu_scroll_dist = 0
held_buttons = []

pyautogui.PAUSE = 0
pydirectinput.PAUSE = 0

def get_cursor_type() -> int:
    """
    @returns
    int id of cursor
    """
    return win32gui.GetCursorInfo()[1]


def reset_buttons():
    global held_buttons
    for button in held_buttons:
        pydirectinput.keyUp(button)

    held_buttons = []


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
    def scroll(gestures):
        reset_buttons()
        global previous_scroll_dist
        left_gesture = gestures[0]
        right_gesture = gestures[1]

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
    def menuscroll(gestures):
        global previous_menu_scroll_dist
        left_gesture = gestures[0]

        left_x, left_y, left_z = left_gesture["landmarks"][HAND_THUMB_END]
        del left_x, left_z

        height_center = int(HEIGHT / 2)

        difference = height_center - left_y
        dist_diff = abs(difference) - abs(previous_menu_scroll_dist)
        if -SCROLL_DIST_DEAD_ZONE < dist_diff < SCROLL_DIST_DEAD_ZONE:
            return True
        
        pydirectinput.moveTo(300, height_center)
        multi = int(abs(dist_diff) // SCROLL_MULTIPLIER)
        multi = max(1, min(multi, 5))
        if difference > previous_menu_scroll_dist:
            [pyautogui.scroll(100) for _ in range(0, multi)]
        if difference < previous_menu_scroll_dist:
            [pyautogui.scroll(-100) for _ in range(0, multi)]
        previous_menu_scroll_dist = difference
        return True

    @staticmethod
    def cursor(gesture):
        global cursor_grab_counter
        reset_buttons()
        position_x, position_y, _ = gesture["landmarks"][HAND_CENTER_POINT]
        del _

        position_x = (position_x / WIDTH) * 1920
        position_y = (position_y / HEIGHT) * 1080
        cursor_position_x, cursor_position_y = pyautogui.position()

        difference = (position_x - cursor_position_x) + (position_y - cursor_position_y)

        x, y = pyautogui.position()
        cursor_type = get_cursor_type()
        if cursor_type == CURSOR_TYPES["auto"]:
            cursor_grab_counter = 0
        if cursor_type == CURSOR_TYPES["move"] and x < 300:
            pydirectinput.leftClick()
        if cursor_type == CURSOR_TYPES["pointer"]:
            if cursor_grab_counter > CURSOR_GRAB:
                cursor_grab_counter = 0
                pydirectinput.leftClick()
            cursor_grab_counter += 1

        # limit cursor movement to min of pixels to limit jitter
        if -CURSOR_DIST_DEAD_ZONE < difference < CURSOR_DIST_DEAD_ZONE:
            return True

        pydirectinput.moveTo(int(position_x), int(position_y))

        return True

    @staticmethod
    def click(gesture):
        reset_buttons()
        pydirectinput.leftClick()
        return False

    @staticmethod
    def right_click(gesture):
        reset_buttons()
        pydirectinput.rightClick()
        return False

    @staticmethod
    def space(gesture):
        reset_buttons()
        pydirectinput.press("m")
        return False

    def backspace(gesture):
        reset_buttons()
        pydirectinput.press("a")
        return False

    @staticmethod
    def _keyboard(button, direction, gestures):
        global held_buttons
        # release all buttons except current
        for btn in held_buttons:
            if btn == button:
                continue
            pydirectinput.keyUp(btn)
            held_buttons.remove(btn)

        cursor_hand = gestures[0]
        if direction == "left":
            cursor_hand = gestures[1]

        if not button in held_buttons:
            pydirectinput.keyDown(button)
            held_buttons.append(button)

        cursor_x, cursor_y, _ = cursor_hand["landmarks"][HAND_CENTER_POINT]
        del _
        width_rect = int(WIDTH / 4)
        height_rect = int(HEIGHT / 3)

        square_x = cursor_x // width_rect
        square_y = cursor_y // height_rect
        buttons = {
            (0, 0): "z",
            (0, 1): "x",
            (0, 2): "c",
            (1, 0): "v",
            (1, 1): "b",
            (1, 2): "n",
            (2, 0): "z",
            (2, 1): "x",
            (2, 2): "c",
            (3, 0): "v",
            (3, 1): "b",
            (3, 2): "n",
        }
        button_id = (square_x, square_y)
        try:
            letter = buttons[button_id]
            pydirectinput.press(letter)
        except KeyError:
            pass
        return True

    @staticmethod
    def input_keyboard(gestures):
        pydirectinput.press("s")
        return False

    @staticmethod
    def left_first_group(gestures):
        return GestureActions._keyboard("e", "left", gestures)

    @staticmethod
    def left_second_group(gestures):
        return GestureActions._keyboard("w", "left", gestures)

    @staticmethod
    def left_third_group(gestures):
        return GestureActions._keyboard("q", "left", gestures)

    @staticmethod
    def right_first_group(gestures):
        return GestureActions._keyboard("r", "right", gestures)

    @staticmethod
    def right_second_group(gestures):
        return GestureActions._keyboard("t", "right", gestures)

    @staticmethod
    def right_third_group(gestures):
        return GestureActions._keyboard("y", "right", gestures)

    @staticmethod
    def nothing(gesture):
        """
        yes i am really gonna leave this here, because it's funny \n
        gosha dembel lives in our code forever
        """
        reset_buttons()
