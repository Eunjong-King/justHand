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
from files.unicode import join_jamos

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

def righthand_function(hand_landmarks, kn):
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
    feature_list = np.round(feature_list, decimals=5)
    C = dic[left_model.predict([feature_list])[0]]

    return C

def lefthand_function(ch, previous_ch, my_word):
    if ch == previous_ch:
        return my_word, ch
    # 무조건 전에 입력한거랑 달라야함 => ㄱ인식됐는데 다음 할게 헷갈리면 ㄱ만 200번 입력되는 대참사 막기 위함
    else:
        # ch가 자음일 때
        if ch in ja:
            # previous_ch도 자음이라서 만약에 합쳐질 수 있는 경우 ex) ㄱ + ㅅ => ㄳ
            for pre, now, res in merge_ja:
                if pre == previous_ch and now == ch:
                    my_word = my_word[:-1] + res
                    return my_word, res
            # 일반적인 경우
            my_word += ch
            return my_word, ch

        # ch가 모음일 때
        elif ch in mo:
            # ㅗ + ㅏ => ㅘ 처럼 합쳐지는 경우
            if previous_ch in mo:
                for pre, now, res in merge_mo:
                    if pre == previous_ch and now == ch:
                        my_word = my_word[:-1] + res
                        return my_word, res
            # ㄱ -> ㅏ -> ㄱ -> ㅅ -> ㅏ 인경우 "갃ㅏ"가 아닌 "각사"가되야함
            elif previous_ch in np.array(merge_ja)[:, 2]:
                for pre, now, res in merge_ja:
                    if res == previous_ch:
                        my_word = my_word[:-1] + pre + now + ch
                        return my_word, ch
            # 아무것도 아닌 경우
            my_word += ch
            return my_word, ch

        # ch가 특수기호일 때
        else:
            # 1일때 쌍자음으로 만들기, 2일때 재입력, 3일때 백스페이스, 4일때 스페이스
            if ch == '1':
                if previous_ch in ssang:
                    # 다행히도 쌍자음의 유니코드는 그냥 자음의 +1이다
                    my_word = my_word[:-1] + chr(ord(previous_ch) + 1)
            elif ch == '2':
                pass
            elif ch == '3':
                my_word = my_word[:-1]
            elif ch == '4':
                my_word += ' '

            return my_word, ch

def click_mouse(right_list):
    label = max(right_list)
    if label == 'z':
        pass
    elif label == 'r':
        mouse_event(mrd, 0, 0)
        mouse_event(mru, 0, 0)
    else:
        mouse_event(mld, 0, 0)
        mouse_event(mlu, 0, 0)

def move_mouse(base, altitude):
    #moveTo(base, altitude)
    SetCursorPos((int(base), int(altitude)))


dot_list = [4, 8, 12, 14, 16, 18, 20]
dic = {'q':'ㅂ', 'w':'ㅈ', 'e':'ㄷ', 'r':'ㄱ', 't':'ㅅ', 'y':'ㅛ', 'u':'ㅕ', 'i':'ㅑ', 'o':'ㅐ', 'p':'ㅔ',
       'a':'ㅁ', 's':'ㄴ', 'd':'ㅇ', 'f':'ㄹ', 'g':'ㅎ', 'h':'ㅗ', 'j':'ㅓ', 'k':'ㅏ', 'l':'ㅣ',
       'z':'ㅋ', 'x':'ㅌ', 'c':'ㅊ', 'v':'ㅍ', 'b':'ㅠ', 'n':'ㅜ', 'm':'ㅡ',
       '1':'1', '2':'2', '3':'3', '4':'4'}
ja = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
mo = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ']
merge_mo = [['ㅗ', 'ㅏ', 'ㅘ'], ['ㅗ', 'ㅐ', 'ㅙ'], ['ㅜ', 'ㅓ', 'ㅝ'], ['ㅜ', 'ㅔ', 'ㅞ']]
ssang = ['ㄱ','ㄷ','ㅂ','ㅅ','ㅈ']
merge_ja = [['ㄱ', 'ㅅ', 'ㄳ'], ['ㄴ', 'ㅈ', 'ㄵ'], ['ㄴ', 'ㅎ', 'ㄶ'], ['ㄹ', 'ㄱ', 'ㄺ'], ['ㄹ', 'ㅁ', 'ㄻ'],
        ['ㄹ', 'ㅂ', 'ㄼ'], ['ㄹ', 'ㅅ', 'ㄽ'], ['ㄹ', 'ㅌ', 'ㄾ'], ['ㄹ', 'ㅍ', 'ㄿ'], ['ㄹ', 'ㅎ', 'ㅀ'],
        ['ㅂ', 'ㅅ', 'ㅄ']]
left_model = joblib.load('files/lefthand_model.pkl')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
WIDTH = GetSystemMetrics(0)
HEIGHT = GetSystemMetrics(1)
print(WIDTH, HEIGHT)
righthand_model = joblib.load('./files/righthand_model.pkl')
cap = cv2.VideoCapture(0)
right_list = []
left_list = []
my_word = ''
previous_ch = ''
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
                if one_hand.classification[0].label == "Right":
                    base, altitude = calculate_vector(hand_landmarks.landmark[5], hand_landmarks.landmark[8])
                    move_mouse(base, altitude)
                    # drawing(image, hand_landmarks)
                    right_list.append(righthand_function(hand_landmarks, righthand_model))
                    if len(right_list) > 10:
                        click_mouse(right_list)
                        right_list = []
                else:
                    C = get_label()
                    left_list.append(C)
                    if len(left_list) > 30:
                        ch = max(left_list)
                        my_word, previous_ch = lefthand_function(ch, previous_ch, my_word)
                        print(join_jamos(my_word))
                        left_list = []

        # 화면 키우는부분 (나중에 이부분 삭제)
        #image = cv2.resize(image, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('mouse and keyboard', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()