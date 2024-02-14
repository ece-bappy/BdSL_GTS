# -*- coding: utf-8 -*-
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "O",
    11: "A",
    12: "I",
    13: "U",
    14: "RA",
    15: "E",
    16: "OI",
    17: "O",
    18: "OU",
    19: "KA",
    20: "KHA",
    21: "GA",
    22: "GHA",
    23: "UMA",
    24: "CHA",
    25: "CHA",
    26: "JA",
    27: "JHA",
    28: "NEO",
    29: "TA",
    30: "THA",
    31: "DA",
    32: "DHA",
    33: "NA",
    34: "TA",
    35: "THA",
    36: "DA",
    37: "DHA",
    38: "PA",
    39: "FA",
    40: "BA",
    41: "MA",
    42: "LA",
    43: "SHA",
    44: "HA",
    45: "ONUSSOR",
    46: "BISORGO",
    47: "CHANDRA BINDU",
}
lable_dict_bangla = {
    0: "০",
    1: "১",
    2: "২",
    3: "৩",
    4: "৪",
    5: "৫",
    6: "৬",
    7: "৭",
    8: "৮",
    9: "৯",
    10: "অ",
    11: "আ",
    12: "ই ঈ",
    13: "উ ঊ",
    14: "ঋ র ড় ঢ়",
    15: "এ",
    16: "ঐ",
    17: "ও",
    18: "ঔ",
    19: "ক",
    20: "খ",
    21: "গ",
    22: "ঘ",
    23: "ঙ",
    24: "চ",
    25: "ছ",
    26: "জ য",
    27: "ঝ",
    28: "ঞ",
    29: "ট",
    30: "ঠ",
    31: "ড",
    32: "ঢ",
    33: "ন ণ",
    34: "ত",
    35: "থ",
    36: "দ",
    37: "ধ",
    38: "প",
    39: "ফ",
    40: "ব ভ",
    41: "ম",
    42: "ল",
    43: "শ ষ স",
    44: "হ",
    45: "হং",
    46: "ঃ",
    47: "ঃ",
}
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

        predicted_character = labels_dict[int(prediction[0])]

        predicted_character_bangla = lable_dict_bangla[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            predicted_character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        print(predicted_character_bangla)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
