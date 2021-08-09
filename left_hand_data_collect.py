import cv2
import mediapipe as mp
import csv
from math import sqrt
import os
from time import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def write_csv(image, key_input):
    label = key_input
    feature_list = []
    mean_x = hand_landmarks.landmark[0].x  # x가 왼오 0이 왼 1이 오
    mean_y = hand_landmarks.landmark[0].y  # y가 위아래 0이 젤위 1이 젤아래
    min_x = w - 1;
    max_x = 0.0;
    min_y = h - 1;
    max_y = 0.0
    for i in range(0, 21):  # 요기부터
        hlm = hand_landmarks.landmark[i]
        if hlm.x * w > max_x:
            max_x = hlm.x * w
        if hlm.x * w < min_x:
            min_x = hlm.x * w
        if hlm.y * h > max_y:
            max_y = hlm.y * h
        if hlm.y * h < min_y:
            min_y = hlm.y * h
    for i in dot_list:
        hlm = hand_landmarks.landmark[i]
        feature_list.append(((hlm.x - mean_x) * w) / (max_x - min_x))
        feature_list.append((hlm.y - mean_y) * h / (max_y - min_y))
    d8 = hand_landmarks.landmark[8]
    d12 = hand_landmarks.landmark[12]
    d16 = hand_landmarks.landmark[16]
    d23 = sqrt((d8.x * w - d12.x * w) ** 2 + (d8.y * h - d12.y * h) ** 2)
    d34 = sqrt((d16.x * w - d12.x * w) ** 2 + (d16.y * h - d12.y * h) ** 2)
    feature_list.append(d23 / d34 - 1)
    feature_list.append((max_y - min_y) / (max_x - min_x) - 1)
    cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 0), 2)
    feature_list.append(label)
    temp_wr.writerow(feature_list)

def write_hangul(image, text):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.text((1, h-35), text, font=font, fill=(0, 0, 255, 0))
    image = np.array(pil_image)
    return image

font = ImageFont.truetype("files/tway_air.ttf", 30)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
dot_list = [4, 8, 12, 14, 16, 18, 20]
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
process_status = 0
dataset_file = open("files/lefthand.csv", 'a', newline='', encoding='utf-8')
dataset_wr = csv.writer(dataset_file)
label_count = 0
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.999,
                    max_num_hands=1
                    ) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        if process_status == 0:
            guild_photo_time = time()
            process_status = 1

        if cv2.waitKey(10) == 27:
            break

        if process_status == 1:
            if time()-guild_photo_time < 2:
                image = cv2.imread("files/guide/"+str(label_count)+".png")
                image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
                image = write_hangul(image, str(label_count)+"번 사진 띄워주는 중...")
            else:
                process_status = 2

        else:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if process_status == 2:
            waiting_time = time()
            process_status = 3



        if process_status == 3:
            now_time = time() - waiting_time
            if 0 <= now_time < 1:
                image = write_hangul(image, str(3) + "초 뒤 데이터화 시작")
            elif 1 <= now_time < 2:
                image = write_hangul(image, str(2) + "초 뒤 데이터화 시작")
            elif 2 <= now_time < 3:
                image = write_hangul(image, str(1) + "초 뒤 데이터화 시작")
            else:
                process_status = 4

        if process_status == 4:
            recording_time = time()
            tempdata_file = open('files/temp_data.csv', 'w', newline='')
            temp_wr = csv.writer(tempdata_file)
            process_status = 5

        if process_status == 5:
            now_time = time() - recording_time
            if now_time < 3:
                if results.multi_hand_landmarks:
                    image = write_hangul(image, "데이터화 중")
                    write_csv(image, label_count)
                    pass
                else:
                    image = write_hangul(image, "손이 안보여요")
            else:
                process_status = 6
                tempdata_file.close()

        if process_status == 6:
            ask_time = time()
            process_status = 7

        if process_status == 7:
            now_time = time() - ask_time
            if now_time < 4:
                image = write_hangul(image, "다시 녹화할거면 'q' / 다음거 갈거면 'e'")
                exit_code = cv2.waitKey(1)
                if exit_code == ord('q'):
                    process_status = 0
                elif exit_code == ord('e'):
                    tempdata_file = open("files/temp_data.csv")
                    reader = csv.reader(tempdata_file)
                    for row in reader:
                        dataset_wr.writerow(row)
                    tempdata_file.close()
                    process_status = 0
                    label_count += 1
            else:
                process_status = 0

        cv2.imshow('dataization', image)

if tempdata_file:
    tempdata_file.close()
if os.path.isfile("files/temp_data.csv"):
    os.remove("files/temp_data.csv")
dataset_file.close()
cap.release()