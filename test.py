import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    min_detection_confidence=0.8, static_image_mode=False, min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)
        primage = cv2.resize(image, (320, 240))
        # Set flag
        primage.flags.writeable = False

        # Detections
        results = hands.process(primage)

        # Set flag to true
        primage.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            print(results.multi_handedness)
            for num, hand in enumerate(results.multi_hand_landmarks):
                # mp_drawing.draw_landmarks(image, hand, mp_hands)
                mp_drawing.draw_landmarks(
                    image,
                    hand,
                    mp_styles.get_default_hand_connections_style(),
                    mp_styles.get_default_hand_landmarks_style(),
                )
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(
                #         color=(255, 255, 255), thickness=2, circle_radius=2
                #     ),
                # )

        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
