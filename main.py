import cv2
import mediapipe as mp
from win32api import GetSystemMetrics, SetCursorPos, mouse_event
from win32con import MOUSEEVENTF_LEFTDOWN as mld
from win32con import MOUSEEVENTF_LEFTUP as mlu
from win32con import MOUSEEVENTF_RIGHTDOWN as mrd
from win32con import MOUSEEVENTF_RIGHTUP as mru
from math import sqrt, atan, sin, cos, tan, degrees
import joblib
import numpy as np
import math
from collections import Counter
import pyautogui
import pyperclip

def calculate_vector(INDEX_MCP, INDEX_TIP):
    # 원공간에서 벡터 (내가 오른쪽 위를 가르키면 (+,+), 왼쪽
    edit_vector = (INDEX_TIP.x - INDEX_MCP.x, INDEX_MCP.y - INDEX_TIP.y)

    # 손가락 벡터의 반원 공간의 반지름
    radius_of_circle = sqrt((INDEX_TIP.x - INDEX_MCP.x)**2
                           +(INDEX_TIP.y - INDEX_MCP.y)**2
                           +(INDEX_TIP.z - INDEX_MCP.z)**2)

    # 반원 공간의 벡터를 xy평면으로 사영
    projection_of_length = sqrt((INDEX_TIP.x - INDEX_MCP.x)**2
                                +(INDEX_TIP.y - INDEX_MCP.y)**2)

    # 사영 했을때 길이와 r의 비율 (직사각형 공간에서 벡터의 길이를 구하기 위함)
    relative_length = projection_of_length / radius_of_circle

    # 기초 값은 다 구했고, 진짜 계산하는 함수 호출
    # 이 함수 전까지는 무조건 잘 됌
    return change_space(edit_vector, relative_length)


def change_space(vector, length):
    # find radian 절대값으로 계산하므로 범위는 (0 ~ ㅠ/2)
    radian = atan(abs(vector[1]/vector[0]))
    # 빗변의 상대적 길이 (단위-pixel) / sin(radian)의 범위는 (0 ~ 1)
    # 옆을 가르키면 radian이 0에 가까워지고 sin(radian)은 0으로 수렴 그럼 hypotenuse가 발산
    # 이 hypotenuse는 대각선 최대값 원에 가까운
    if degrees(radian) > degrees(atan(HEIGHT/WIDTH)):
        hypotenuse = ((HEIGHT/2) / sin(radian))
    else:
        hypotenuse = ((WIDTH/2) / cos(radian))
    # 빗변의 절대적 길이
    hypotenuse = hypotenuse * length
    # 밑변, 높이 길이의 절대값
    base = cos(radian) * hypotenuse
    altitude = tan(radian) * base

    base = base if vector[0] > 0 else -1*base
    altitude = altitude if vector[1] > 0 else -1*altitude

    base += (WIDTH/2)
    altitude += (HEIGHT/2)
    altitude = HEIGHT - altitude
    return base, altitude

def drawing(image, hand_landmarks):
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    tip_x = int(640 * hand_landmarks.landmark[8].x)
    tip_y = int(480 * hand_landmarks.landmark[8].y)
    mcp_x = int(640 * hand_landmarks.landmark[5].x)
    mcp_y = int(480 * hand_landmarks.landmark[5].y)
    # cv2.line(image, (tip_x, tip_y), (tip_x, tip_y), (0, 0, 255), 10)
    # cv2.line(image, (mcp_x, mcp_y), (mcp_x, mcp_y), (0, 0, 255), 10)
    cv2.arrowedLine(image, (mcp_x, mcp_y), (tip_x, tip_y), (0,0,255), thickness=10, tipLength=0.2)

def calculate_distance(dot_a, dot_b):
    pow_x = (dot_a.x - dot_b.x) ** 2
    pow_y = (dot_a.y - dot_b.y) ** 2
    return sqrt(pow_x + pow_y)

def lefthand_function(hand_landmarks, kn):
    distance_4_12 = calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[4])
    distance_16_20 = calculate_distance(hand_landmarks.landmark[16], hand_landmarks.landmark[20])
    features = [distance_4_12, distance_16_20]
    return kn.predict([features])[0]

def get_label():
    feature_list = []
    mean_x = hand_landmarks.landmark[0].x  # x가 왼오 0이 왼 1이 오
    mean_y = hand_landmarks.landmark[0].y  # y가 위아래 0이 젤위 1이 젤아래
    min_x = WIDTH - 1;
    max_x = 0.0;
    min_y = HEIGHT - 1;
    max_y = 0.0
    for i in range(0, 21):  # 요기부터
        hlm = hand_landmarks.landmark[i]
        if hlm.x * WIDTH > max_x:
            max_x = hlm.x * WIDTH
        if hlm.x * WIDTH < min_x:
            min_x = hlm.x * WIDTH
        if hlm.y * HEIGHT > max_y:
            max_y = hlm.y * HEIGHT
        if hlm.y * HEIGHT < min_y:
            min_y = hlm.y * HEIGHT
    for i in dot_list:
        hlm = hand_landmarks.landmark[i]
        feature_list.append(((hlm.x - mean_x) * WIDTH) / (max_x - min_x))
        feature_list.append((hlm.y - mean_y) * HEIGHT / (max_y - min_y))
    d8 = hand_landmarks.landmark[8]
    d12 = hand_landmarks.landmark[12]
    d16 = hand_landmarks.landmark[16]
    d23 = math.sqrt((d8.x * WIDTH - d12.x * WIDTH) ** 2 + (d8.y * HEIGHT - d12.y * HEIGHT) ** 2)
    d34 = math.sqrt((d16.x * WIDTH - d12.x * WIDTH) ** 2 + (d16.y * HEIGHT - d12.y * HEIGHT) ** 2)
    feature_list.append(d23 / d34 - 1)
    feature_list.append((max_y - min_y) / (max_x - min_x) - 1)
    feature_list = np.round(feature_list, decimals=5)
    C = label_char[righthand_model.predict([feature_list])[0]]

    return C

def righthand_function(ch, previous_ch):
        # ch가 자음일 때
        if ch in ja or ch in mo:
            pyperclip.copy(ch)
            pyautogui.hotkey('ctrl', 'v')
            return ch

        # ch가 특수기호일 때
        else:
            # 1일때 쌍자음으로 만들기, 2일때 재입력, 3일때 백스페이스, 4일때 스페이스
            if ch == '1':
                if previous_ch in ssang:
                    # 다행히도 쌍자음의 유니코드는 그냥 자음의 +1이다
                    pyautogui.press('backspace')
                    pyperclip.copy(chr(ord(previous_ch) + 1))
                    pyautogui.hotkey('ctrl', 'v')
            elif ch == '2':
                pass
            elif ch == '3':
                pyautogui.press('backspace')
            elif ch == '4':
                pyautogui.press(' ')

            return ch

def click_mouse(left_list):
    label = Counter(left_list).most_common()[0][0]
    if label == 'z':
        pass
    elif label == 'r':
        mouse_event(mrd, 0, 0)
        mouse_event(mru, 0, 0)
    else:
        mouse_event(mld, 0, 0)
        mouse_event(mlu, 0, 0)

def move_mouse(base, altitude):
    SetCursorPos((int(base), int(altitude)))


dot_list = [4, 8, 12, 14, 16, 18, 20]
label_char = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ',
              'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ',
              '1','2','3','4']
ja = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
mo = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ']
ssang = ['ㄱ','ㄷ','ㅂ','ㅅ','ㅈ']
righthand_model = joblib.load('right/Right-model.pkl')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
WIDTH = GetSystemMetrics(0)
HEIGHT = GetSystemMetrics(1)
lefthand_model = joblib.load('left/Left-model.pkl')
cap = cv2.VideoCapture(0)
right_list = []
left_list = []
previous_ch = ''
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.999,
                    max_num_hands=2
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
                if one_hand.classification[0].label == "Left":
                    base, altitude = calculate_vector(hand_landmarks.landmark[5], hand_landmarks.landmark[8])
                    move_mouse(base, altitude)
                    left_list.append(lefthand_function(hand_landmarks, lefthand_model))
                    if len(left_list) > 10:
                        click_mouse(left_list)
                        left_list = []
                if one_hand.classification[0].label == "Right":
                    C = get_label()
                    if previous_ch != C:
                        right_list.append(C)
                        if len(right_list) > 30:
                            ch = Counter(right_list).most_common()[0][0]
                            previous_ch = righthand_function(ch, previous_ch)
                            right_list = []

        cv2.imshow('mouse and keyboard', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()