import csv
import argparse
import itertools
from mss import mss
from mouse import move
from copy import deepcopy
from keyboard import send
from utils import CvFpsCalc
import pynput.keyboard as kb
from mediapipe import solutions
from flask.wrappers import Response
from model import KeyPointClassifier
from flask import Flask, render_template
from pynput.mouse import Controller, Button
from numpy import asarray, where, array,empty,append
from config import (
    WIDTH,
    HEIGHT,
    VERTICAL_FLIP,
    COUNT_OF_FINGER_POINT,
    BLACK,
    WHITE,
    PHONE,
    IMAGE_PATH
)
from cv2 import (
    waitKey,
    VideoCapture,
    resizeWindow,
    resize,
    rectangle,
    putText,
    namedWindow,
    line,
    LINE_AA,
    inRange,
    imread,
    imencode,
    getTrackbarPos,
    FONT_HERSHEY_SIMPLEX,
    flip,
    cvtColor,
    createTrackbar,
    COLOR_RGB2BGR,
    COLOR_GRAY2BGR,
    COLOR_BGR2RGB,
    COLOR_BGR2HSV,
    circle,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    boundingRect,
    bitwise_and,
    addWeighted,
)

class Camera:
    __slots__ = ["cap_width","cap_height","cap","img","green","frame","hsv","lower_green","upper_green","mask","f","l_h","l_s","l_v","u_h","u_s","u_v","state","width","height"]
    
    def __init__(self):
        self.cap_width, self.cap_height = WIDTH, HEIGHT
        self.cap = VideoCapture(-1)
        # PHONE
        self.cap.set(CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(CAP_PROP_FRAME_HEIGHT,self.cap_height)

    def capture_camera(self):
        _, self.frame = self.cap.read()
        return self.frame
    
    def flip(self):
        return flip(self.frame,VERTICAL_FLIP)
    
    def run(self):
        self.frame = self.capture_camera()
        self.frame = self.flip()
        return self.frame


class GreenScreen:
    def __init__(self):
        self.state = 0
        self.img = resize(imread(IMAGE_PATH),(WIDTH,HEIGHT))

    def nothing(self, one):
        pass

    def cvt_to_hsv(self):
        return cvtColor(self.frame,COLOR_BGR2HSV)
    
    def find_range(self):
        return inRange(self.hsv,self.lower_green,self.upper_green)
    
    def bitwise(self):
        return bitwise_and(self.frame,self.frame,mask=self.mask)
    
    def cvt_to_bgr(self):
        return cvtColor(self.mask,COLOR_GRAY2BGR)
    
    def find_where_pixels_is_equal_to_mask(self):
        return where(self.mask,self.img,self.f)
    
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
        if self.state == 0:
            self.trackbar()
        self.l_h,self.l_s,self.l_v,self.u_h,self.u_s,self.u_v = self.values()
        self.lower_green = array([self.l_h, self.l_s, self.l_v])
        self.upper_green = array([self.u_h, self.u_s, self.u_v])
        self.hsv = self.cvt_to_hsv()
        self.mask = self.find_range()
        self.f = self.frame - self.bitwise()
        self.mask = self.cvt_to_bgr()
        self.frame = self.find_where_pixels_is_equal_to_mask()
        if waitKey(1) & 0xFF == ord('q'):
            pass
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
                if handedness.classification[0].label[0:] == "Left":
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
                            case "RightClick":
                                self.mouse.click(Button.right)
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
            landmark_x2 = min(int(landmark.x * WIDTH), WIDTH - 1)
            landmark_y2 = min(int(landmark.y * HEIGHT), HEIGHT - 1)

            landmark_point.append([landmark_x, landmark_y])
            landmark_point_x.append(landmark_x2)
            landmark_point_y.append(landmark_y2)
        move(sum(landmark_point_x)/COUNT_OF_FINGER_POINT,sum(landmark_point_y)/COUNT_OF_FINGER_POINT)
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


    def draw_info_text(self,image, brect, handedness, hand_sign_text,
                    ):
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


class Screen:
    __slots__ = ["width_for_capture", "height_for_capture", "bounding","width_for_all_cameras","height_for_all_cameras","frame"]

    def __init__(self):
        self.bounding = {
            "top": 0,
            "left": 0,
            "width": WIDTH,
            "height": HEIGHT,
        }

    def capture_screen(self):
        return mss().grab(self.bounding)
    
    def convert_frame_to_array(self):
        return asarray(self.frame)
    
    def run(self):
        self.frame = self.capture_screen()
        self.frame = self.convert_frame_to_array()
        return self.frame


class Merge:
    __slots__ = ["frame","hands", "news", "imgs", "imgs1", "news1", "news2"]

    def convert_to_array(self) -> tuple:
        return asarray(self.news), asarray(self.imgs)
    
    def convert_color(self):
        return cvtColor(self.news, COLOR_RGB2BGR)
    
    def convert_color2(self):
        return cvtColor(self.news, COLOR_BGR2RGB)

    def merge(self): 
        return addWeighted(self.news, 0.5, self.imgs, 1, 0)
    
    def run(self, imgs, news) -> tuple:
        self.imgs = imgs
        self.news = news
        self.news, self.imgs = self.convert_to_array()
        self.news = self.convert_color()
        self.news = self.convert_color2()
        self.frame = self.merge()
        return self.frame


class Starter:
    def __init__(self):
        self.camera = Camera()
        self.screen = Screen()
        self.merge = Merge()
        self.gesture = Gesture()
        self.greenscreen = GreenScreen()
    
    def run(self):
        while 1:
            camera = self.camera.run()
            camera = self.gesture.run(camera)
            camera = self.greenscreen.run(camera)
            screen = self.screen.run()
            merge = self.merge.run(camera,screen)
            return merge


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
 