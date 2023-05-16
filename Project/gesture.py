import csv
import itertools
from mediapipe import solutions
from model import KeyPointClassifier
from pynput.mouse import Controller, Button
from utils import CvFpsCalc
from cv2 import (
    waitKey,
    rectangle,
    putText,
    line,
    LINE_AA,
    cvtColor,
    COLOR_BGR2RGB,
    circle,
    boundingRect,
    FONT_HERSHEY_SIMPLEX
)
from config import (
    WIDTH,
    HEIGHT,
    COUNT_OF_FINGER_POINTS,
    BLACK,
    WHITE,
)
from copy import deepcopy
import pynput.keyboard as kb
import argparse
from keyboard import send
from numpy import array,empty,append
from mouse import move

class Gesture:
    def __init__(self) -> None:
        args = self.get_args()
        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence
        
        self.use_brect = True

        mp_hands = solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.mouse = Controller()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.keyboard = kb.Controller()
        self.key = kb.Key
        self.state = 0
        self.state2 = 0

    def cvt_color(self):
        return cvtColor(self.image,COLOR_BGR2RGB)

    def switch_flags_state(self, state):
        self.image.flags.writeable = state

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--use_static_image_mode', action='store_true')
        parser.add_argument("--min_detection_confidence",
                            help='min_detection_confidence',
                            type=float,
                            default=0.7)
        parser.add_argument("--min_tracking_confidence",
                            help='min_tracking_confidence',
                            type=int,
                            default=0.5)

        args = parser.parse_args()

        return args

    def select_mode(self,key, mode):
        number = 6
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        return number, mode

    def calc_bounding_rect(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [array((landmark_x, landmark_y))]

            landmark_array = append(landmark_array, landmark_point, axis=0)

        x, y, w, h = boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []
        landmark_point_x = []
        landmark_point_y = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            # Mouse
            landmark_x2 = min(int(landmark.x * WIDTH), WIDTH - 1)
            landmark_y2 = min(int(landmark.y * HEIGHT), HEIGHT - 1)

            landmark_point.append([landmark_x, landmark_y])
            landmark_point_x.append(landmark_x2)
            landmark_point_y.append(landmark_y2)
        move(sum(landmark_point_x)/COUNT_OF_FINGER_POINTS,sum(landmark_point_y)/COUNT_OF_FINGER_POINTS)
        return landmark_point

    def pre_process_landmark(self,landmark_list):
        temp_landmark_list = deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    def logging_csv(self,number, mode, landmark_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = './model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        return

    def draw_landmarks(self,image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
         BLACK, 6)
            line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
         WHITE, 2)
            line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
         BLACK, 6)
            line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    WHITE, 2)

            # Index finger
            line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
         BLACK, 6)
            line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
         WHITE, 2)
            line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
         BLACK, 6)
            line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
         WHITE, 2)
            line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
         BLACK, 6)
            line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
         WHITE,6) 
            # Middle finger
            line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
         BLACK, 6)
            line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
         WHITE, 2)
            line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
         BLACK, 6)
            line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
         WHITE, 2)
            line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
         BLACK, 6)
            line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
         WHITE,6) 
            # Ring finger
            line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
         BLACK, 6)
            line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
         WHITE, 2)
            line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
         BLACK, 6)
            line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
         WHITE, 2)
            line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
         BLACK, 6)
            line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
         WHITE, 2)

            # Little finger
            line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
         BLACK, 6)
            line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
         WHITE, 2)
            line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
         BLACK, 6)
            line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
         WHITE, 2)
            line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
         BLACK, 6)
            line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    WHITE, 2)

            # Palm
            line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
         BLACK, 6)
            line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
         WHITE, 2)
            line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
         BLACK, 6)
            line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
         WHITE, 2)
            line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
         BLACK, 6)
            line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
         WHITE, 2)
            line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
         BLACK, 6)
            line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
         WHITE, 2)
            line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
         BLACK, 6)
            line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
         WHITE, 2)
            line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
         BLACK, 6)
            line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
         WHITE, 2)
            line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
         BLACK, 6)
            line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    WHITE, 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 1:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 2:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 3:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 4:  
                circle(image, (landmark[0], landmark[1]), 8, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 5:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 6:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 7:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 8:  
                circle(image, (landmark[0], landmark[1]), 8, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 9:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 10:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 11:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 12:  
                circle(image, (landmark[0], landmark[1]), 8, WHITE,
                        -1)
                circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 13:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 14:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 15:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 16:  
                circle(image, (landmark[0], landmark[1]), 8, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
            if index == 17:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 18:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 19:  
                circle(image, (landmark[0], landmark[1]), 5, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
            if index == 20:  
                circle(image, (landmark[0], landmark[1]), 8, WHITE,
             -1)
                circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)

        return image

    def draw_bounding_rect(self,use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        BLACK, 1)

        return image

    def draw_info_text(self,image, brect, handedness, hand_sign_text):
        rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    BLACK, -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, LINE_AA)

        return image

    def draw_info(self,image, fps, mode, number):
        putText(image, "FPS:" + str(fps), (10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, BLACK, 4, LINE_AA)
        putText(image, "FPS:" + str(fps), (10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, WHITE, 2, LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                    FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1,
                    LINE_AA)
            if 0 <= number <= 9:
                putText(image, "NUM:" + str(number), (10, 110),
                        FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1,
                        LINE_AA)
        return image
    
    def run(self, image):
        self.image = image
        self.mode = 0
        fps = self.cvFpsCalc.get()
        key = waitKey(10)
        if key == 27:  # ESC
            pass
        number, mode = self.select_mode(key, self.mode)
        debug_image = deepcopy(image)
        # Detection implementation #############################################################
        self.image = self.cvt_color()
        self.switch_flags_state(False)
        results = self.hands.process(image)
        self.switch_flags_state(True)
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                # Write to the dataset file
                self.logging_csv(number, mode, pre_processed_landmark_list)
                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                # Drawing part
                debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(debug_image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id],
                )
                if handedness.classification[0].label[0:] == "Right":
                    if self.state == 0:
                        match self.keypoint_classifier_labels[hand_sign_id]:
                            case "FirstGroup":
                                send("q")
                                print("q")
                                self.state = 1
                            case "SecondGroup":
                                send("w")
                                print("w")
                                self.state = 1
                            case "ThirdGroup":
                                send("e")
                                print("e")
                                self.state = 1
                            case "FirstLetter":
                                send("z")
                                print("z")
                                self.state = 1
                            case "SecondLetter":
                                send("x")
                                print("x")
                                self.state = 1
                            case "ThirdLetter":
                                send("c")
                                print("c")
                                self.state = 1
                            case "Click":
                                self.mouse.click(Button.left)
                                self.state = 1
                            case "RightClick":
                                self.mouse.click(Button.right)
                                self.state = 1
                    if self.keypoint_classifier_labels[hand_sign_id] == "CloseNothing":
                        self.state = 0
                else:
                    if self.state2 == 0:
                        match self.keypoint_classifier_labels[hand_sign_id]:
                            case "FirstGroup":
                                send("r")
                                print("r")
                                self.state2 = 1
                            case "SecondGroup":
                                send("t")
                                print("t")
                                self.state2 = 1
                            case "ThirdGroup":
                                send("y")
                                print("y")
                                self.state2 = 1
                            case "FirstLetter":
                                send("v")
                                print("v")
                                self.state2 = 1
                            case "SecondLetter":
                                send("b")
                                print("b")
                                self.state2 = 1
                            case "ThirdLetter":
                                send("n")
                                print("n")
                                self.state2 = 1
                            case "Click":
                                self.mouse.click(Button.left)
                                self.state2 = 1
                            case "RightClick":
                                self.mouse.click(Button.right)
                                self.state2 = 1
                    if self.keypoint_classifier_labels[hand_sign_id] == "CloseNothing":
                        self.state2 = 0
        debug_image = self.draw_info(debug_image, fps, mode, number)
        # Screen reflection #############################################################
        return debug_image