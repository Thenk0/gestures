import math
import itertools
from copy import deepcopy
from mediapipe import solutions
from numpy import array
from .gestureActions import GestureActions
from model import KeyPointClassifier
from .bufferlessCapture import BufferlessVideoCapture
from .gestureNames import GestureNames
from cv2 import (
    resize,
    cvtColor,
    COLOR_BGR2RGB,
    line,
    circle,
    boundingRect,
    rectangle,
    FONT_HERSHEY_SIMPLEX,
    LINE_AA,
    putText,
    imshow,
    waitKey,
)
from config import (
    FLIP_HANDS,
)


def getProgress(start, current, finish):
    duration = finish - start
    progress = finish - current
    return progress / duration


def easeInOutSine(t):
    return -(math.cos(math.pi * t) - 1) / 2


class Gestures:
    def __init__(self):
        self.hand_recognition = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            model_complexity=0,
        )
        # in frames
        self.color_animation_duration = 15
        self.current_right_color = (255, 255, 255)
        self.current_left_color = (255, 255, 255)
        self.timer = 0
        self.render_image_size = (0, 0)
        self.gesture_classifier = KeyPointClassifier()
        self.gesture_actions = GestureActions()

    def _get_points_color():
        pass

    def draw_hands(
        self,
        frame,
        hand,
        default_points_color=(0, 0, 0),
        bone_lines_color=(0, 0, 0),
    ):
        """
        Draws hands on provided screen with specified colors

        Parameters:
        -----------------
        frame : np_array
            frame to draw hand on
        hand : list
            a list of hand point coordinates, the list is not normalized
        default_points_color: tuple, optional
            color to draw hand points
        bone_lines_color: tuple, optional
            color to draw hand lines
        """
        for connection in solutions.hands.HAND_CONNECTIONS:
            start, end = connection
            line_size = min(
                8, max(4, int((hand[start][2] * -40 + hand[end][2] * -40) / 2))
            )
            start = tuple([hand[start][0], hand[start][1]])
            end = tuple([hand[end][0], hand[end][1]])
            line(
                frame,
                start,
                end,
                (bone_lines_color),
                line_size + 4,
            )
            line(
                frame,
                start,
                end,
                (0,0,0),
                line_size,
            )
        for point in hand:
            circle_size = min(15, max(5, int(point[2] * -40)))
            circle(frame, (point[0], point[1]), circle_size, default_points_color, -1)
            circle(frame, (point[0], point[1]), circle_size, bone_lines_color, 1)

        return frame

    def draw_info(self, frame, hand):
        """
        draws information about gestures on given frame

        Parameters:
        ----------------

        frame: np_array
            a frame to draw info to
        hand: dict
            custom dict type
        """

        x, y, w, h = hand["bounding_rect"]
        rectangle(
            frame,
            (x, y),
            (w, h),
            (0, 0, 0),
            1,
        )
        rectangle(
            frame,
            (x, y),
            (w, y - 22),
            (0, 0, 0),
            -1,
        )
        info_text = f"{hand['label']} : {hand['gesture']}"
        putText(
            frame,
            info_text,
            (x + 5, y - 4),
            FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            LINE_AA,
        )
        return frame

    def calc_landmark_list(self, hands):
        """
        converts mediapipe coordinates to absolute coordinates of render_image

        Parameters:
        --------------
        hands : NamedTuple
            mediapipe multihand_landmark type
        """
        landmark_point = []
        image_width, image_height = self.render_image_size
        for _, landmark in enumerate(hands):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y, landmark.z])
        return landmark_point

    def pre_process_landmark(self, landmarks):
        """
        some landmark thingamajig to do something
        """
        temp_landmark_list = deepcopy(landmarks)
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
        max_value = max(list(map(abs, temp_landmark_list)))

        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
        return temp_landmark_list

    def recognize_gesture(self, hand_landmarks) -> str:
        landmarks = self.pre_process_landmark(hand_landmarks)
        gesture_id = self.gesture_classifier(landmarks)
        gesture = GestureNames.gestureNames[gesture_id]
        return gesture

    def action(self, gesture_list, gesture_count) -> None:
        if gesture_count == 1:
            self._single_hand_action(gesture_list[0])
        if gesture_count == 2:
            self._multi_hand_action(gesture_list)

    def _single_hand_action(self, gesture) -> None:
        try:
            method = GestureNames.gestureActionsSingle[gesture["gesture"]]
            repeat = method(gesture)
        except:
            GestureActions.nothing()

    def _multi_hand_action(self, gestures) -> None:
        try:
            gesture_string = f"{gestures[0]['gesture']}|{gestures[1]['gesture']}"
            method = GestureNames.gestureActionsMultiple[gesture_string]
            repeat = method(gestures)
        except:
            GestureActions.nothing()

    def _get_gesture_list(self, hand_gestures) -> list:
        """
        Parses and adapts mediapipe hands to a more simple format

        Parameters:
        ----------------
        hand_gestures : NamedTuple
            mediapipes gestures tuple with multiple hands
        """
        gesture_list = []
        for index, hand_landmarks in enumerate(hand_gestures.multi_hand_landmarks):
            hand_label = hand_gestures.multi_handedness[index].classification[0].label
            # change hands around, because mediapipe is weird
            if FLIP_HANDS:
                hand_label = "Right" if hand_label == "Left" else "Left"
            landmark_list = self.calc_landmark_list(hand_landmarks.landmark)

            # remove z from list to calculate bounding rect and gesture id
            landmark_array = []
            for landmark in landmark_list:
                landmark_array.append([landmark[0], landmark[1]])

            landmark_array = array(landmark_array)
            x, y, w, h = boundingRect(landmark_array)
            gesture_list.append(
                {
                    "label": hand_label,
                    "landmarks": landmark_list,
                    "gesture": self.recognize_gesture(landmark_array),
                    "bounding_rect": (x, y, x + w, y + h),
                }
            )
        return gesture_list

    def run(self, image, screen=None):
        """
        Runs gesture recognition and draws results on screen.
        This method takes control of keyboard and mouse on use

        Parameters:
        ----------------
        image : np_array
            cv2 image with hands
        screen : np_array, optional
            by default is None, if screen is provided, gestures will be drawn here

        Returns:
        ----------------
        Image with drawn hands and info
        """
        render_image = image if screen is None else screen

        self.render_image_size = (render_image.shape[1], render_image.shape[0])

        processed_image = cvtColor(image, COLOR_BGR2RGB)
        processed_image = resize(processed_image, (300, 300))
        processed_image.flags.writeable = False

        hand_gestures = self.hand_recognition.process(processed_image)
        del processed_image  # delete image to save memory

        if hand_gestures.multi_hand_landmarks is None:
            return render_image

        gesture_list = self._get_gesture_list(hand_gestures)
        gesture_count = len(gesture_list)

        if gesture_count == 0:
            return render_image
        for gesture in gesture_list:
            render_image = self.draw_hands(
                render_image,
                gesture["landmarks"],
                default_points_color=GestureNames.gestureColors[gesture["gesture"]],
            )
            render_image = self.draw_info(render_image, gesture)
        self.action(gesture_list, len(gesture_list))
        self.timer += 1

        return render_image

    @staticmethod
    def test():
        cap = BufferlessVideoCapture(1, False)
        screenCap = BufferlessVideoCapture(2, True)
        gestures = Gestures()
        while True:
            frame = cap.read()
            screen = screenCap.read()
            render = gestures.run(frame, screen)
            imshow("Hand Tracking", render)
            if waitKey(10) & 0xFF == ord("q"):
                break
  