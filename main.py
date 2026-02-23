import cv2
import numpy as np
import mediapipe as mp
from math import sqrt, pow

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

dots = ((8, 6),   # указательный
        (12, 10), # средний
        (16, 14), # безымянный
        (20, 18)) # мизинец

previous_state = None
cap = cv2.VideoCapture(0)

def get_coords(landmarks, tip, pip):
    tip_coord = (int(landmarks[tip].x * w), int(landmarks[tip].y * h))
    pip_coord = (int(landmarks[pip].x * w), int(landmarks[pip].y * h))
    
    return tip_coord, pip_coord

def get_dist(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    c1 = pow(x1 - x0, 2)
    c2 = pow(y1 - y0, 2)

    return sqrt(c1+c2)

while cap.isOpened():
    status, frame = cap.read()
    h, w, _ = frame.shape

    current_state = None

    if not status:
        break

    if cv2.waitKey(20) == 27:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks is not None:
        fingers_closed = [] 

        for hand_landmark, hand_nandedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            landmarks = hand_landmark.landmark

            for dot in dots:
                tip_y, pip_y = [i[1] for i in get_coords(landmarks, dot[0], dot[1])]
                fingers_closed.append(tip_y > pip_y)  # y=0 вверху

        current_state = all(fingers_closed)

        if previous_state == False and current_state == True:
            print('рука закрылась')

        previous_state = current_state
        fingers_closed.clear()
         
    else:
        is_closed = None
        print('РУКА НЕ ОБНАРУЖЕНА')
    

    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()