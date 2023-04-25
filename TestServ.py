import csv
import argparse
import itertools

from utils import CvFpsCalc
from model import KeyPointClassifier


from mss import mss
import pynput.keyboard as kb
from mediapipe import solutions
from keyboard import is_pressed,send,press
from flask.wrappers import Response
from mouse import move, get_position
from flask import Flask, render_template
from pynput.mouse import Controller, Button
from numpy import asarray, where, array, float32, argmax,empty,append
from copy import deepcopy
from cv2 import (
    boundingRect,
    putText,
    FONT_HERSHEY_SIMPLEX,
    LINE_AA,
    INTER_AREA,
    rectangle,
    inRange,
    line,
    resize,
    VideoCapture,
    flip,
    cvtColor,
    COLOR_BGR2RGB,
    addWeighted,
    imencode,
    bitwise_and,
    FILLED,
    circle,
    COLOR_RGB2BGR,
    COLOR_BGR2HSV,
    COLOR_GRAY2BGR,
    getTrackbarPos,
    namedWindow,
    resizeWindow,
    createTrackbar,
    imread,
    waitKey,
    destroyAllWindows,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    imshow
)

class Camera:
    __slots__ = ["cap_width","cap_height","cap","img","green","frame","hsv","lower_green","upper_green","mask","f","l_h","l_s","l_v","u_h","u_s","u_v","state","width","height"]
    
    def __init__(self):
        self.cap_width, self.cap_height = 1280, 720
        self.cap = VideoCapture(-1)
        # http://192.168.31.107:8080/video
        self.cap.set(CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(CAP_PROP_FRAME_HEIGHT,self.cap_height)

    def capture_camera(self):
        _, self.frame = self.cap.read()
        return self.frame
    
    def convert_frame_to_array(self):
        return asarray(self.frame)
    
    def resize_frame(self):
        return resize(self.frame,(self.cap_width, self.cap_height))

    def flip(self):
        return flip(self.frame,-1)
    
    def run(self):
        self.frame = self.capture_camera()
        self.frame = self.convert_frame_to_array()
        self.frame = self.resize_frame()
        self.frame = self.flip()
        return self.frame


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
            max_num_hands=2,
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


    def run(self, image):
        self.mode = 0
        fps = self.cvFpsCalc.get()
        key = waitKey(10)
        if key == 27:  # ESC
            pass
        number, mode = self.select_mode(key, self.mode)
        debug_image = deepcopy(image)
        # Detection implementation #############################################################
        image = cvtColor(image, COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(
                    landmark_list)
                # Write to the dataset file
                self.logging_csv(number, mode, pre_processed_landmark_list)
                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
        
            
                # Drawing part
                debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                )
                if handedness.classification[0].label[0:] == "Left" and self.state == 0:
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
                        case "RightClick":
                            self.mouse.click(Button.right)
                    if self.keypoint_classifier_labels[hand_sign_id] == "CloseNothing":
                        self.state = 0
                else:
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


    def select_mode(self,key, mode):
        number = 8
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
            landmark_x2 = min(int(landmark.x * 1920), 1920 - 1)
            landmark_y2 = min(int(landmark.y * 1080), 1080- 1)

            landmark_point.append([landmark_x, landmark_y])
            landmark_point_x.append(landmark_x2)
            landmark_point_y.append(landmark_y2)
        move(sum(landmark_point_x)/21,sum(landmark_point_y)/21)
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
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
         (255, 255, 255),6) 
            # Middle finger
            line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
         (255, 255, 255),6) 
            # Ring finger
            line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
         (255, 255, 255), 2)

            # Little finger
            line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
         (255, 255, 255), 2)
            line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
         (0, 0, 0), 6)
            line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
             -1)
                circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image


    def draw_bounding_rect(self,use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 0, 0), 1)

        return image


    def draw_info_text(self,image, brect, handedness, hand_sign_text,
                    ):
        rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, LINE_AA)

        return image


    def draw_info(self,image, fps, mode, number):
        putText(image, "FPS:" + str(fps), (10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, LINE_AA)
        putText(image, "FPS:" + str(fps), (10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                    FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    LINE_AA)
            if 0 <= number <= 9:
                putText(image, "NUM:" + str(number), (10, 110),
                        FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        LINE_AA)
        return image


class Screen:
    __slots__ = ["width_for_capture", "height_for_capture", "bounding","width_for_all_cameras","height_for_all_cameras","frame"]

    def __init__(self):
        self.width_for_capture, self.height_for_capture = 1280, 720
        self.width_for_all_cameras, self.height_for_all_cameras = 1280, 720
        self.bounding = {
            "top": 0,
            "left": 0,
            "width": self.width_for_capture,
            "height": self.height_for_capture,
        }

    def capture_screen(self):
        return mss().grab(self.bounding)
    
    def convert_frame_to_array(self):
        return asarray(self.frame)
    
    def resize_frame(self):
        return resize(self.frame,(self.width_for_all_cameras, self.height_for_all_cameras))

    def run(self):
        self.frame = self.capture_screen()
        self.frame = self.convert_frame_to_array()
        self.frame = self.resize_frame()
        return self.frame


class Merge:
    __slots__ = ["hands", "news", "imgs", "imgs1", "news1", "news2"]

    def convert_to_array(self) -> tuple:
        return asarray(self.news), asarray(self.imgs)
    
    def convert_color(self):
        return cvtColor(self.news, COLOR_RGB2BGR)

    def merge(self): 
        return addWeighted(self.news, 1, self.imgs, 0.8, 1)

    def run(self, imgs, news) -> tuple:
        self.imgs = imgs
        self.news = news
        self.news, self.imgs = self.convert_to_array()
        self.news = self.convert_color()
        return self.merge()


class Starter:
    def __init__(self):
        self.camera = Camera()
        self.screen = Screen()
        self.merge = Merge()
        self.gesture = Gesture()
    
    def run(self):
        while 1:
            camera = self.camera.run()
            camera = self.gesture.run(camera)
            screen = self.screen.run()
            merge = self.merge.run(camera,screen)
            return camera


starter = Starter()
app = Flask(__name__)


def get_frames():
    while 1:
        frame = starter.run()
        return frame


frame1 = None


def record_into_global_frame():
    while 1:
        global frame1
        frame1 = get_frames()
        _, buffer = imencode(".jpg", asarray(frame1))
        frame = buffer.tobytes()
        yield frame


def record_into_global_frame_for_second_thread():
    while 1:
        global frame1
        frame2 = deepcopy(frame1)
        _, buffer = imencode(".jpg", asarray(frame2))
        frame = buffer.tobytes()
        yield frame


def redirection_of_threads():
    while 1:
        frametest1 = record_into_global_frame()
        frametest2 = record_into_global_frame_for_second_thread()
        return frametest1, frametest2


def yeil():
        frametest1,_ = asarray(redirection_of_threads())
        for i in frametest1:
            yield b"--frame\r\n" b"Content-Type: image/jpg\r\n\r\n" + i + b"\r\n"


def yeil2():
        _, frametest2 = redirection_of_threads()
        for i in frametest2:
            yield b"--frame\r\n" b"Content-Type: image/jpg\r\n\r\n" + i + b"\r\n"
 
@app.route("/video_feed")
def video_feed():
        return Response(yeil(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/news")
def vids():
        return Response(yeil2(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=False)
 