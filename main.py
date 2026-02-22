import cv2
import numpy as np
import mediapipe as mp
from math import sqrt, pow

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

def get_dist(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    c1 = pow(x1 - x0, 2)
    c2 = pow(y1 - y0, 2)

    return sqrt(c1+c2)

while cap.isOpened():
    status, frame = cap.read()
    h, w, _ = frame.shape

    if not status:
        break

    if cv2.waitKey(20) == 27:
        break
    
    frame = cv2.flip(frame, 1)
    framw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks is not None:
        for hand_landmark, hand_nandedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            landmarks = hand_landmark.landmark

            index_tip_finger = (int(landmarks[8].x * w), int(landmarks[8].y * h))
            index_mcp_finger = (int(landmarks[5].x * w), int(landmarks[5].y * h))
            index_dist = get_dist(index_tip_finger, index_mcp_finger)

            print(index_dist)

            cv2.circle(frame, index_tip_finger, 3, (0, 0, 255), -1)
            cv2.circle(frame, index_mcp_finger, 3, (0, 0, 255), -1)

    else:
        print('not found')
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()