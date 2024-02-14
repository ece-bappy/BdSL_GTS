# -*- coding: utf-8 -*-
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from gtts import gTTS
import os

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)
# Set the webcam resolution
cap.set(4, 400)  # Width
cap.set(3, 300)  # Height

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.5
)

labels_dict = {0: "0", 1: "1", 2: "2"}
lable_dict_bangla = {0: "র", 1: "আ", 2: "উ"}
image_folder = "imgL"  # The folder containing corresponding images

# Initialize a deque to store the last three detected images and the last detected letter
detected_images = deque(maxlen=3)
last_detected_letter = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_letter = labels_dict[int(prediction[0])]
        predicted_character_bangla = lable_dict_bangla[int(prediction[0])]

        # Load and display corresponding image only if a different letter is detected
        if predicted_letter != last_detected_letter:
            tts = gTTS(predicted_character_bangla, lang="bn")
            tts.save("character.mp3")

            # Play the audio
            os.system("mpg123 character.mp3")

        last_detected_letter = predicted_letter

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
