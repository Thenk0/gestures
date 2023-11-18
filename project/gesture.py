import csv
import itertools
from mediapipe import solutions
from model import KeyPointClassifier
import pyautogui
import math
from pynput.mouse import Controller, Button
from google.protobuf.json_format import MessageToDict
from cv2 import (
    resize,
    putText,
    waitKey,
    rectangle,
    line,
    LINE_AA,
    cvtColor,
    COLOR_BGR2RGB,
    circle,
    boundingRect,
    FONT_HERSHEY_SIMPLEX,
    FILLED,
)
from config import (
    WIDTH,
    GESTURE_ACCURACY,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    FLIP_HANDS,
)
from copy import deepcopy
import pynput.keyboard as kb
from keyboard import send
from numpy import array, empty, append, asarray
from mouse import move


class Gesture:
    def __init__(self):
        # Write gestures
        self.mode = False
        self.number = 10
        self.gesture_lock = False
        self.use_brect = True
        mp_hands = solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.mouse = Controller()
        with open(
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        self.keyboard = kb.Controller()
        self.key = kb.Key
        self.gesture = 0
        self.gesture_repeat = 0
        self.scroll_point = [0, 0]
        self.gestures_for_left_hand = {
            "FirstGroup": "q",
            "SecondGroup": "w",
            "ThirdGroup": "e",
            "FirstLetter": "z",
            "SecondLetter": "x",
            "ThirdLetter": "c",
            "Click": "nothing",
            "RightClick": "a",
            "CloseNothing": "nothing",
            "BACK": "a",
        }
        self.gestures_for_right_hand = {
            "FirstGroup": "r",
            "SecondGroup": "t",
            "ThirdGroup": "y",
            "FirstLetter": "v",
            "SecondLetter": "b",
            "ThirdLetter": "n",
            "Click": "LClick",
            "RightClick": "RClick",
            "CloseNothing": "nothing",
            "BACK": "scroll",
        }
        self.dist = WIDTH
        self.dist2 = 0
        self.act = 0

    def cvt_frame_color(self):
        return cvtColor(self.image, COLOR_BGR2RGB)

    def switch_flags_state(self, state):
        self.image.flags.writeable = state

    def select_mode(self):
        if 48 <= self.key <= 57:  # 0 ~ 9
            self.number = self.key - 48
        if self.key == 110:  # n
            print("stopped gesture writing")
            self.mode = False
        if self.key == 107:  # k
            print("started gesture writing")
            self.mode = True

    def get_image_shape(self, image):
        return self.image.shape[1], self.image.shape[0]

    def calc_bounding_rect(self):
        landmark_array = empty((0, 2), int)
        for _, landmark in enumerate(self.hand_landmarks.landmark):
            landmark_x = min(int(landmark.x * self.image_width), self.image_width - 1)
            landmark_y = min(int(landmark.y * self.image_height), self.image_height - 1)

            landmark_point = asarray([array((landmark_x, landmark_y))])
            landmark_array = asarray(append(landmark_array, landmark_point, axis=0))

        x, y, w, h = boundingRect(landmark_array)
        return array((x, y, x + w, y + h))

    def calc_landmark_list(self):
        landmark_point = []
        for _, landmark in enumerate(self.hand_landmarks.landmark):
            landmark_x = min(int(landmark.x * self.image_width), self.image_width - 1)
            landmark_y = min(int(landmark.y * self.image_height), self.image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return array(landmark_point)

    def move_mouse(self, move_to):
        x, y = move_to
        move(x, y)

    def draw_dot(self):
        x, y = pyautogui.position()
        scale = WIDTH / SCREEN_WIDTH
        x = round(x * scale)
        y = round(y * scale)

        radius = 5
        self.image = circle(
            self.image,
            (x - (radius // 2), y - (radius // 2)),
            radius,
            (0, 0, 255),
            FILLED,
        )
        return self.image

    def pre_process_landmark(self):
        temp_landmark_list = deepcopy(self.get_landmark_list)
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)) )

        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
        return temp_landmark_list

    def logging_csv(self):
        if not self.mode:
            return
        if self.mode and (0 <= self.number <= 9):
            csv_path = "./model/keypoint_classifier/keypoint.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.number, *self.pre_processed_landmark_list])

    def draw_landmarks(self):
        if len(self.get_landmark_list) > 0:
            # Thumb
            line(
                self.image,
                tuple(self.get_landmark_list[2]),
                tuple(self.get_landmark_list[3]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[2]),
                tuple(self.get_landmark_list[3]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[3]),
                tuple(self.get_landmark_list[4]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[3]),
                tuple(self.get_landmark_list[4]),
                WHITE,
                2,
            )

            # Index finger
            line(
                self.image,
                tuple(self.get_landmark_list[5]),
                tuple(self.get_landmark_list[6]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[5]),
                tuple(self.get_landmark_list[6]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[6]),
                tuple(self.get_landmark_list[7]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[6]),
                tuple(self.get_landmark_list[7]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[7]),
                tuple(self.get_landmark_list[8]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[7]),
                tuple(self.get_landmark_list[8]),
                WHITE,
                6,
            )
            # Middle finger
            line(
                self.image,
                tuple(self.get_landmark_list[9]),
                tuple(self.get_landmark_list[10]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[9]),
                tuple(self.get_landmark_list[10]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[10]),
                tuple(self.get_landmark_list[11]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[10]),
                tuple(self.get_landmark_list[11]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[11]),
                tuple(self.get_landmark_list[12]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[11]),
                tuple(self.get_landmark_list[12]),
                WHITE,
                6,
            )
            # Ring finger
            line(
                self.image,
                tuple(self.get_landmark_list[13]),
                tuple(self.get_landmark_list[14]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[13]),
                tuple(self.get_landmark_list[14]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[14]),
                tuple(self.get_landmark_list[15]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[14]),
                tuple(self.get_landmark_list[15]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[15]),
                tuple(self.get_landmark_list[16]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[15]),
                tuple(self.get_landmark_list[16]),
                WHITE,
                2,
            )

            # Little finger
            line(
                self.image,
                tuple(self.get_landmark_list[17]),
                tuple(self.get_landmark_list[18]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[17]),
                tuple(self.get_landmark_list[18]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[18]),
                tuple(self.get_landmark_list[19]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[18]),
                tuple(self.get_landmark_list[19]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[19]),
                tuple(self.get_landmark_list[20]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[19]),
                tuple(self.get_landmark_list[20]),
                WHITE,
                2,
            )

            # Palm
            line(
                self.image,
                tuple(self.get_landmark_list[0]),
                tuple(self.get_landmark_list[1]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[0]),
                tuple(self.get_landmark_list[1]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[1]),
                tuple(self.get_landmark_list[2]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[1]),
                tuple(self.get_landmark_list[2]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[2]),
                tuple(self.get_landmark_list[5]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[2]),
                tuple(self.get_landmark_list[5]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[5]),
                tuple(self.get_landmark_list[9]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[5]),
                tuple(self.get_landmark_list[9]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[9]),
                tuple(self.get_landmark_list[13]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[9]),
                tuple(self.get_landmark_list[13]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[13]),
                tuple(self.get_landmark_list[17]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[13]),
                tuple(self.get_landmark_list[17]),
                WHITE,
                2,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[17]),
                tuple(self.get_landmark_list[0]),
                BLACK,
                6,
            )
            line(
                self.image,
                tuple(self.get_landmark_list[17]),
                tuple(self.get_landmark_list[0]),
                WHITE,
                2,
            )

        # Key Points
        for index, landmark in enumerate(self.get_landmark_list):
            if index == 0:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 1:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 2:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 3:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 4:
                circle(self.image, (landmark[0], landmark[1]), 8, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 5:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 6:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 7:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 8:
                circle(self.image, (landmark[0], landmark[1]), 8, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 9:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 10:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 11:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 12:
                circle(self.image, (landmark[0], landmark[1]), 8, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 13:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 14:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 15:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 16:
                circle(self.image, (landmark[0], landmark[1]), 8, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 17:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 18:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 19:
                circle(self.image, (landmark[0], landmark[1]), 5, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 20:
                circle(self.image, (landmark[0], landmark[1]), 8, WHITE, -1)
                circle(self.image, (landmark[0], landmark[1]), 8, BLACK, 1)

        return self.image

    def draw_bounding_rect(self):
        if self.use_brect:
            rectangle(
                self.image,
                (self.brect[0], self.brect[1]),
                (self.brect[2], self.brect[3]),
                BLACK,
                1,
            )
        return self.image

    def draw_info_text(self):
        rectangle(
            self.image,
            (self.brect[0], self.brect[1]),
            (self.brect[2], self.brect[1] - 22),
            BLACK,
            -1,
        )
        info_text = self.handedness.classification[0].label
        if FLIP_HANDS:
            info_text = "Right" if info_text == "Left" else "Left"
        if self.hand_sign_text != "":
            info_text = info_text + ":" + self.hand_sign_text
        putText(
            self.image,
            info_text,
            (self.brect[0] + 5, self.brect[1] - 4),
            FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            1,
            LINE_AA,
        )
        return self.image

    def draw_info(self):
        mode_string = ["Logging Key Point", "Logging Point History"]
        if 1 <= self.mode <= 2:
            putText(
                self.image,
                "MODE:" + mode_string[self.mode - 1],
                (10, 90),
                FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                1,
                LINE_AA,
            )
            if 0 <= self.number <= 9:
                putText(
                    self.image,
                    "NUM:" + str(self.number),
                    (10, 110),
                    FONT_HERSHEY_SIMPLEX,
                    0.6,
                    WHITE,
                    1,
                    LINE_AA,
                )
        return self.image

    def hands_action(self, hand, hand_sign):
        if self.gesture != hand_sign:
            self.gesture = hand_sign
            self.gesture_lock = False
            self.gesture_repeat == 0
            return

        self.gesture_repeat += 1
        if self.gesture_repeat < GESTURE_ACCURACY:
            return

        action = ""
        if hand == "Right":
            action = self.gestures_for_right_hand[hand_sign]
        if hand == "Left":
            action = self.gestures_for_left_hand[hand_sign]

        if self.gesture_lock and action != "scroll":
            return
        # if length is 1, assume that user click buttons
        if len(action) == 1:
            send(action)
        if action == "scroll":
            if self.scroll_point[0] > self.mouse_x:
                self.mouse.scroll(0, -1)
            else:
                self.mouse.scroll(0, 1)
            self.scroll_point = [self.mouse_x, self.mouse_y]
        if action == "RClick":
            self.mouse.click(Button.right)
        if action == "LClick":
            self.mouse.click(Button.left)
        self.gesture_lock = True

        # if self.hand_sign_text == self.keypoint_classifier_labels[8]:
        #     if self.mouse_x < self.dist and self.mouse_y > self.dist2:
        #         self.dist = self.mouse_x
        #         self.dist2 = self.mouse_y
        #         self.mouse.scroll(1,1)
        #     if self.mouse_x > self.dist and self.mouse_y < self.dist2:
        #         self.dist = self.mouse_x
        #         self.dist2 = self.mouse_y
        #         self.mouse.scroll(-1,-1)

    def run(self, image, do_gestures=True):
        self.image = image

        self.key = waitKey(1)
        self.select_mode()
        debug_image = cvtColor(image, COLOR_BGR2RGB)
        debug_image = resize(debug_image, (300, 300))
        self.switch_flags_state(False)
        results = self.hands.process(debug_image)
        self.switch_flags_state(True)
        if results.multi_hand_landmarks is None:
            self.image = self.draw_info()
            return self.image
        for self.hand_landmarks, self.handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            hand = self.handedness.classification[0].label
            # change hands around, because mediapipe is weird
            if FLIP_HANDS:
                hand = "Right" if hand == "Left" else "Left"
            self.image_width, self.image_height = self.get_image_shape()

            # Calculate mouse position
            self.mouse_x = min(
                int(self.hand_landmarks.landmark[9].x * SCREEN_WIDTH), SCREEN_WIDTH - 1
            )
            self.mouse_y = min(
                int(self.hand_landmarks.landmark[9].y * SCREEN_HEIGHT),
                SCREEN_HEIGHT - 1,
            )

            self.get_landmark_list = self.calc_landmark_list()  # Landmark calculation
            self.brect = self.calc_bounding_rect()  # Bounding box calculation
            self.image = self.draw_bounding_rect()
            self.image = self.draw_landmarks()

            if not do_gestures:
                return self.image
            if not self.act and hand == "Right":
                self.move_mouse((self.mouse_x, self.mouse_y))
                self.image = self.draw_dot()  # Draw dot for mouse
            self.pre_processed_landmark_list = (
                self.pre_process_landmark()
            )  # Conversion to relative coordinates / normalized coordinates
            # self.logging_csv() # Write to the dataset file
            hand_sign_id = self.keypoint_classifier(
                self.pre_processed_landmark_list
            )  # Hand sign classification
            # Drawing part
            self.hand_sign_text = self.keypoint_classifier_labels[hand_sign_id]
            self.image = self.draw_info_text()
            self.hands_action(hand, self.hand_sign_text)
        return self.image
