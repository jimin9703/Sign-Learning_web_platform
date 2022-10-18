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

global pr_index
app.config['lang_code'] = ['en', 'ko']
max_num_hands = 1
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

model = load_model('model/handtrain(500(88.2)).h5')
e_model = load_model('model/E_handtrain(400(91)).h5')
quiz_path = "C:\\Users\\dudwh\\signlanguage1\\Individual_img/*.jpg"
jpg_list = [f for f in glob.glob(quiz_path)]

# crop_img 엉키지 않게
class Microsecond(object):
    def __init__(self):
        dt = datetime.now()
        self.microsecond = dt.microsecond

    
    def get_path_name(self):
        return 'model/' + str(self.microsecond)

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
    k_list = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ",
    "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"
]
    return k_list

def get_E_list():
    E_list = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def k_list_idx(element):
    next_topic = ""
    previous_topic = ""

    k_list = get_k_list()

    list_idx_end = len(k_list) - 1  # 마지막 인덱스
    idx_now = k_list.index(element)

    if idx_now == list_idx_end:
        next_topic = k_list[0]
    else:
        next_topic = k_list[idx_now + 1]

    if idx_now != 0:
        previous_topic = k_list[idx_now - 1]

    return next_topic, previous_topic

def Egen(camera):
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 30
        out = cv2.VideoWriter('evideo.avi', fourcc, fps, (int(width), int(height)))

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

                cv2.putText(img, text=get_Elabel(idx).upper(),
                            org=(300,120),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=2)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        out.write(img)
        if cv2.waitKey(1) == ord('q'):
            break

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
                draw.text(org, text, font=font, fill=(0, 0, 0))
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        out.write(img)
        if cv2.waitKey(1) == ord('q'):
            break

def problem(camera):
    camera = cv2.VideoCapture(0)
    k_list = random.sample(get_k_list(), 5)
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
                img = Image.fromarray(img)

                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (300, 50)
                text = get_label(idx)
                draw.text(org, text, font=font, fill=(0, 0, 0))
                img = np.array(img)

                pr_index = text
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # out.write(img)
        # if cv2.waitKey(1) == ord('q'):
        #     break

# def

# # 퀴즈 결과
# @app.route('/quiz/result')
# def result():

# video streaming
@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/E_video_feed')
def E_video_feed():
    camera = cv2.VideoCapture(0)
    return Response(Egen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/quiz')
def quiz():
    camera = cv2.VideoCapture(0)
    return Response(problem(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/d_study')
def d_study():
    return render_template('d_study.html', link=request.full_path)

@app.route('/d_test')
def d_test():
    return render_template('d_test.html', link=request.full_path)

@app.route('/index')
def index():
    return render_template('index.html', link=request.full_path)

@app.route('/j_study')
def j_study():
    return render_template('j_study.html', link=request.full_path)

@app.route('/j_test')
def j_test():
    k_list = random.sample(get_k_list(), 5)
    return render_template('j_test.html', k_list=k_list, link=request.full_path)

@app.route('/Ej_study')
def Ej_study():
    return render_template('Ej_study.html', link=request.full_path)

@app.route('/Ej_test')
def Ej_test():
    E_list = random.sample(get_E_list(), 5)
    return render_template('Ej_test.html', E_list=E_list, link=request.full_path)

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