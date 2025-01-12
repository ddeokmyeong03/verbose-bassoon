import cv2
import datetime
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox, Button
import webbrowser
import os

emotion_labels = ['Happy', 'Sad', 'Angry', 'Surprised']  # 기쁨, 슬픔, 분노, 당황에 대응

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def get_emotion_probabilities(prediction):
    return {emotion: float(prob) for emotion, prob in zip(emotion_labels, prediction)}

# 우울증 지수 계산 함수
def calculate_depression_index(emotion_probs):
    joy_prob = emotion_probs['Happy']
    sad_prob = emotion_probs['Sad']
    angry_prob = emotion_probs['Angry']
    surprised_prob = emotion_probs['Surprised']
    depression_index = (0.5 * sad_prob) + (0.3 * angry_prob) + (0.2 * surprised_prob) - (0.5 * joy_prob)
    return max(0, depression_index)

survey_url = "https://forms.gle/LEMXufxAsgoTPpgc6"

# 우울증 지수가 60 이상일 때 GUI 경고창 띄우기 함수
def show_warning():
    root = tk.Tk()
    root.title("경고")

    # 경고 메시지
    label = tk.Label(root, text="우울증 지수가 60 이상입니다. 2차 검사를 실시합니다.", padx=20, pady=10)
    label.pack()

    # 돌아가기 버튼
    def on_close():
        root.destroy()

    back_button = Button(root, text="돌아가기", command=on_close)
    back_button.pack(side="left", padx=10, pady=10)

    # 설문조사 버튼
    def open_survey():
        webbrowser.open(survey_url)

    survey_button = Button(root, text="설문조사", command=open_survey)
    survey_button.pack(side="right", padx=10, pady=10)

    root.mainloop()

def main():
    # Haar Cascade 로드
    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # 감정 분석 모델 로드
    emotion_model = load_model('model.keras')

    # 카메라 설정
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    emotion_labels = ['Happy', 'Sad', 'Angry', 'Surprised']  # 기쁨, 슬픔, 분노, 당황에 대응

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, (96, 96))
            face_resized = face_resized.reshape(1, 96, 96, 1).astype('float32') / 255

            # 감정 예측 및 확률 딕셔너리 변환
            emotion_prediction = emotion_model.predict(face_resized)
            emotion_probabilities = get_emotion_probabilities(emotion_prediction[0])

            # 우울증 지수 계산
            depression_index = calculate_depression_index(emotion_probabilities)
            emotion_label = max(emotion_probabilities, key=emotion_probabilities.get)

            # 우울증 지수에 따른 박스 색상 설정
            if depression_index >= 0.60:
                box_color = (0, 0, 255)  # 빨간색
                show_warning()  # 경고 메시지 GUI 표시
            elif depression_index >= 0.50:
                box_color = (0, 140, 255)  # 주황색
            elif depression_index >= 0.40:
                box_color = (0, 255, 255)  # 노란색
            else:
                box_color = (255, 255, 255)  # 흰색

            # 각 감정 확률과 우울증 지수 문자열 생성
            emotion_text = f"Emotion: {emotion_label}"
            probability_text = ', '.join([f"{emotion}: {prob:.2f}" for emotion, prob in emotion_probabilities.items()])
            depression_text = f"Depression Index: {depression_index:.2f}"

            # 결과 출력
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(img, emotion_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.putText(img, probability_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            cv2.putText(img, depression_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        cv2.imshow('Emotion Detector', img)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()