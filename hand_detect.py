import cv2
import mediapipe as mp
from win32api import GetSystemMetrics, SetCursorPos, mouse_event
from win32con import MOUSEEVENTF_LEFTDOWN as mld
from win32con import MOUSEEVENTF_LEFTUP as mlu
from win32con import MOUSEEVENTF_RIGHTDOWN as mrd
from win32con import MOUSEEVENTF_RIGHTUP as mru
from math import sqrt, atan, sin, cos, tan, degrees
import joblib

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
    print(kn.predict([features]))

def move_mouse(base, altitude):
    #moveTo(base, altitude)
    SetCursorPos((int(base), int(altitude)))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
WIDTH = GetSystemMetrics(0)
HEIGHT = GetSystemMetrics(1)
print(WIDTH, HEIGHT)
righthand_model = joblib.load('./files/righthand_model.pkl')
cap = cv2.VideoCapture(0)
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
                base, altitude = calculate_vector(hand_landmarks.landmark[5], hand_landmarks.landmark[8])
                move_mouse(base, altitude)
                # print(one_hand.classification[0].label, end=" // ")
                # drawing(image, hand_landmarks)
                righthand_function(hand_landmarks, righthand_model)
        # 화면 키우는부분 (나중에 이부분 삭제)
        #image = cv2.resize(image, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('mouse and keyboard', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()