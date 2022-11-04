import os
import time
import glob
import errno
import json
import random
import cv2
from flask import Flask, url_for, render_template, Response, request, redirect, g, session
from keras.utils.image_utils import img_to_array, load_img
from keras.models import load_model
from collections import Counter
import numpy as np
from flask_babel import Babel, gettext
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

app = Flask(__name__)
babel = Babel()
babel.init_app(app)
pr_index = []
pr_index2 = []
pr_index3 = []
pr_index4 = []
pr_index5 = []

Epr_index = []
Epr_index2 = []
Epr_index3 = []
Epr_index4 = []
Epr_index5 = []

try_problem = []
Etry_problem = []

ki = 5
Ei = 5
global alp_list
global k_list
app.config['lang_code'] = ['en', 'ko']
max_num_hands = 1
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

model = load_model('model/handtrain(400(92)).h5')
e_model = load_model('model/E_handtrain(400(91)).h5')
quiz_path = "C:\\Users\\dudwh\\signlanguage1\\Individual_img/*.jpg"
jpg_list = [f for f in glob.glob(quiz_path)]

# h5 모델 
def get_label(idx):
    label = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ",
    "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"
]
    return label[idx]

def get_Elabel(idx):
    Elabel = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]
    return Elabel[idx]


# ** 전역변수 대신 클래스 객체 사용
class PredictLabel(object):
    def __init__(self, label):
        self.label = label

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

class EPredictLabel(object):
    def __init__(self, Elabel):
        self.Elabel = Elabel

    def set_Elabel(self, Elabel):
        self.Elabel = Elabel

    def get_Elabel(self):
        return self.Elabel

predict_label = PredictLabel('')
Epredict_label = EPredictLabel('')

@app.before_request
def before_request():
    g.total_q = 5

def get_k_list():
    ko_list = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ","ㅈ",
               "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ","ㅑ", "ㅓ",
               "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"]
    return ko_list

k_list = random.sample(get_k_list(), 5)

def get_E_list():
    Ei_list = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return Ei_list

alp_list = random.sample(get_E_list(), 5)

def Egen(camera):
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = e_model.predict([data]).squeeze()
                idx = np.argmax(result)

                img = Image.fromarray(img)

                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (300, 50)
                text = get_Elabel(idx)
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))

                img = np.array(img)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 30
        out = cv2.VideoWriter('video.avi', fourcc, fps, (int(width), int(height)))

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],:]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)

                img = Image.fromarray(img)

                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (300,50)
                text = get_label(idx)
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))

                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def problem(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        camera.set(cv2.CAP_PROP_POS_MSEC, 30)

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)
                global pr_index
                pr_index.append(get_label(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                prr_index = get_label(idx)
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(pr_index) == 60):
            k_result = "".join(k_list[0])
            answer = max(set(pr_index), key=pr_index.count)
            global ki
            if (k_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (200, 50)
                text = "정답!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                ki -= 1
                global try_problem
                try_problem.append(k_list[0])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "오답ㅠㅠ"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(pr_index) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def problem2(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        camera.set(cv2.CAP_PROP_POS_MSEC, 30)

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)
                global pr_index2
                pr_index2.append(get_label(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(pr_index2) == 60):
            k_result = "".join(k_list[1])
            answer = max(set(pr_index2), key=pr_index2.count)
            global ki
            if (k_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (200, 50)
                text = "정답!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                ki -= 1
                global try_problem
                try_problem.append(k_list[1])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "오답ㅠㅠ"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(pr_index2) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def problem3(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        camera.set(cv2.CAP_PROP_POS_MSEC, 30)

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)
                global pr_index3
                pr_index3.append(get_label(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(pr_index3) == 60):
            k_result = "".join(k_list[2])
            answer = max(set(pr_index3), key=pr_index3.count)
            global ki
            if (k_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (200, 50)
                text = "정답!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                ki -= 1
                global try_problem
                try_problem.append(k_list[2])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "오답ㅠㅠ"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(pr_index3) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def problem4(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        camera.set(cv2.CAP_PROP_POS_MSEC, 30)

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)
                global pr_index4
                pr_index4.append(get_label(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(pr_index4) == 60):
            k_result = "".join(k_list[3])
            answer = max(set(pr_index4), key=pr_index4.count)
            global ki
            if (k_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (200, 50)
                text = "정답!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                ki -= 1
                global try_problem
                try_problem.append(k_list[3])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "오답ㅠㅠ"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(pr_index4) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def problem5(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        camera.set(cv2.CAP_PROP_POS_MSEC, 30)

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)
                global pr_index5
                pr_index5.append(get_label(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(pr_index5) == 60):
            k_result = "".join(k_list[4])
            answer = max(set(pr_index5), key=pr_index5.count)
            global ki
            if (k_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (200, 50)
                text = "정답!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                ki -= 1
                global try_problem
                try_problem.append(k_list[4])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "오답ㅠㅠ"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(pr_index5) > 60):
            print(ki)
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def Eproblem(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = e_model.predict([data]).squeeze()
                idx = np.argmax(result)

                global Epr_index
                Epr_index.append(get_Elabel(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(Epr_index) == 60):
            alp_result = "".join(alp_list[0])
            answer = max(set(Epr_index), key=Epr_index.count)
            global Ei
            if (alp_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (140, 50)
                text = "correct!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                Ei -= 1
                global Etry_problem
                Etry_problem.append(alp_list[0])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "Whoops!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(Epr_index) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def Eproblem2(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = e_model.predict([data]).squeeze()
                idx = np.argmax(result)

                global Epr_index2
                Epr_index2.append(get_Elabel(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(Epr_index2) == 60):
            alp_result = "".join(alp_list[1])
            answer = max(set(Epr_index2), key=Epr_index2.count)
            global Ei
            if (alp_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (140, 50)
                text = "correct!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                Ei -= 1
                global Etry_problem
                Etry_problem.append(alp_list[1])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "Whoops!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(Epr_index2) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def Eproblem3(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = e_model.predict([data]).squeeze()
                idx = np.argmax(result)

                global Epr_index3
                Epr_index3.append(get_Elabel(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(Epr_index3) == 60):
            alp_result = "".join(alp_list[2])
            answer = max(set(Epr_index3), key=Epr_index3.count)
            global Ei
            if (alp_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (140, 50)
                text = "correct!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                Ei -= 1
                global Etry_problem
                Etry_problem.append(alp_list[2])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "Whoops!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(Epr_index3) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def Eproblem4(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = e_model.predict([data]).squeeze()
                idx = np.argmax(result)

                global Epr_index4
                Epr_index4.append(get_Elabel(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(Epr_index4) == 60):
            alp_result = "".join(alp_list[3])
            answer = max(set(Epr_index4), key=Epr_index4.count)
            global Ei
            if (alp_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (140, 50)
                text = "correct!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                Ei -= 1
                global Etry_problem
                Etry_problem.append(alp_list[3])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "Whoops!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(Epr_index4) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def Eproblem5(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = e_model.predict([data]).squeeze()
                idx = np.argmax(result)

                global Epr_index5
                Epr_index5.append(get_Elabel(idx))  # pr_index ㄱㄴㄷ !! 들이 fps=30 으로 담겨져 있습니다.
                img = Image.fromarray(img)
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if (len(Epr_index5) == 60):
            alp_result = "".join(alp_list[0])
            answer = max(set(Epr_index5), key=Epr_index5.count)
            global Ei
            if (alp_result == answer):
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (140, 50)
                text = "correct!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='⭕', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
            else:
                Ei -= 1
                global Etry_problem
                Etry_problem.append(alp_list[4])
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (120, 50)
                text = "Whoops!"
                draw.text(org, text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
                img = np.array(img)
                # cv2.putText(img, text='❌', org=(200, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                #             color=(0, 0, 0), thickness=3)
        elif (len(Epr_index5) > 60):
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# video streaming
@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/E_video_feed')
def E_video_feed():
    camera = cv2.VideoCapture(0)
    return Response(Egen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jquiz')
def jquiz():
    camera = cv2.VideoCapture(0)
    return Response(problem(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jquiz2')
def jquiz2():
    camera = cv2.VideoCapture(0)
    return Response(problem2(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jquiz3')
def jquiz3():
    camera = cv2.VideoCapture(0)
    return Response(problem3(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jquiz4')
def jquiz4():
    camera = cv2.VideoCapture(0)
    return Response(problem4(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jquiz5')
def jquiz5():
    camera = cv2.VideoCapture(0)
    return Response(problem5(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Ejquiz')
def Ejquiz():
    camera = cv2.VideoCapture(0)
    return Response(Eproblem(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Ejquiz2')
def Ejquiz2():
    camera = cv2.VideoCapture(0)
    return Response(Eproblem2(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Ejquiz3')
def Ejquiz3():
    camera = cv2.VideoCapture(0)
    return Response(Eproblem3(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Ejquiz4')
def Ejquiz4():
    camera = cv2.VideoCapture(0)
    return Response(Eproblem4(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Ejquiz5')
def Ejquiz5():
    camera = cv2.VideoCapture(0)
    return Response(Eproblem5(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index')
def index():
    return render_template('index.html', link=request.full_path)

@app.route('/j_study')
def j_study():
    return render_template('j_study.html', link=request.full_path)

@app.route('/j_test')
def j_test():
    return render_template('j_test.html', k_list=k_list, link=request.full_path)

@app.route('/j_test2')
def j_test2():
    return render_template('j_test2.html', k_list=k_list, link=request.full_path)

@app.route('/j_test3')
def j_test3():
    return render_template('j_test3.html', k_list=k_list, link=request.full_path)

@app.route('/j_test4')
def j_test4():
    return render_template('j_test4.html', k_list=k_list, link=request.full_path)

@app.route('/j_test5')
def j_test5():
    return render_template('j_test5.html', k_list=k_list, link=request.full_path)

@app.route('/Ej_test')
def Ej_test():
    return render_template('Ej_test.html', alp_list=alp_list, link=request.full_path)

@app.route('/Ej_test2')
def Ej_test2():
    return render_template('Ej_test2.html', alp_list=alp_list, link=request.full_path)

@app.route('/Ej_test3')
def Ej_test3():
    return render_template('Ej_test3.html', alp_list=alp_list, link=request.full_path)

@app.route('/Ej_test4')
def Ej_test4():
    return render_template('Ej_test4.html', alp_list=alp_list, link=request.full_path)

@app.route('/Ej_test5')
def Ej_test5():
    return render_template('Ej_test5.html', alp_list=alp_list, link=request.full_path)

@app.route('/result')
def result():
    if ki == 1:
        img_path = "static/img/20.jpg"
    elif ki == 2:
        img_path = "static/img/40.jpg"
    elif ki == 3:
        img_path = "static/img/60.jpg"
    elif ki == 4:
        img_path = "static/img/80.jpg"
    elif ki == 5:
        img_path = "static/img/100.jpg"
    else:
        img_path = "static/img/0.jpg"
    total = ki*20
    return render_template('result.html', ki=ki, total=total, try_problem=try_problem, img_path=img_path, link=request.full_path)

@app.route('/Eresult')
def Eresult():
    if Ei == 1:
        img_path = "static/img/20.jpg"
    elif Ei == 2:
        img_path = "static/img/40.jpg"
    elif Ei == 3:
        img_path = "static/img/60.jpg"
    elif Ei == 4:
        img_path = "static/img/80.jpg"
    elif Ei == 5:
        img_path = "static/img/100.jpg"
    else:
        img_path = "static/img/0.jpg"
    total = Ei*20
    return render_template('Eresult.html', Ei=Ei, total=total, Etry_problem=Etry_problem, img_path=img_path, link=request.full_path)

@app.route('/Ej_study')
def Ej_study():
    return render_template('Ej_study.html', link=request.full_path)

@app.route('/programmer')
def programmer():
    return render_template('programmer.html', link=request.full_path)

@app.route('/inquiry')
def inquiry():
    return render_template('inquiry.html', link=request.full_path)

@app.route('/kr_main')
def kr_main():
    return render_template('kr_main.html', link=request.full_path)

@app.route('/EN_main')
def EN_main():
    return render_template('EN_main.html', link=request.full_path)

@app.route('/')
def main():
    return render_template('main.html', link=request.full_path)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5000, debug=True)