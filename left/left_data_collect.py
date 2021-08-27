import cv2
import mediapipe as mp
import csv
from math import sqrt
import keyboard

def calculate_distance(dot_a, dot_b):
    pow_x = (dot_a.x - dot_b.x) ** 2
    pow_y = (dot_a.y - dot_b.y) ** 2
    return sqrt(pow_x + pow_y)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
file = open('./lefthand.csv', 'a', newline='')
wr = csv.writer(file)
is_recording = False
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.999,
                    max_num_hands=1
                    ) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            two_hands = results.multi_handedness
            # 손이 2개 이상일 때, one_hand와 hand_landmarks가 같은 손을 return함
            for one_hand, hand_landmarks in zip(two_hands, results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                key_input = cv2.waitKey(1)
                if key_input == ord('r'):
                    distance_4_12 = calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[4])
                    distance_16_20 = calculate_distance(hand_landmarks.landmark[16], hand_landmarks.landmark[20])
                    features = [distance_4_12, distance_16_20, "r"]
                    cv2.putText(image, "right click...", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=3)
                    wr.writerow(features)
                elif key_input == ord('l'):
                    distance_4_12 = calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[4])
                    distance_16_20 = calculate_distance(hand_landmarks.landmark[16], hand_landmarks.landmark[20])
                    features = [distance_4_12, distance_16_20, "l"]
                    cv2.putText(image, "left click...", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=3)
                    wr.writerow(features)
                elif key_input == ord('z'):
                    distance_4_12 = calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[4])
                    distance_16_20 = calculate_distance(hand_landmarks.landmark[16], hand_landmarks.landmark[20])
                    features = [distance_4_12, distance_16_20, "z"]
                    cv2.putText(image, "no click...", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=3)
                    wr.writerow(features)
                else:
                    cv2.putText(image, "'l' to left, 'r' to right, 'z' to standard", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=3)

        cv2.imshow('mouse and keyboard', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

file.close()
cap.release()