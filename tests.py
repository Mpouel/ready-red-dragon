import cv2
import keyboard
import time
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = 'faces2.mp4'  # Replace with your test video filename
if not os.path.exists(video_path):
    raise FileNotFoundError("Test video not found. Upload a file named 'faces.mp4' to workspace.")

cap = cv2.VideoCapture(video_path)

last_trigger_time = 0
trigger_cooldown = 5  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 2 and (time.time() - last_trigger_time > trigger_cooldown):
        print("Teacher detected !")
        last_trigger_time = time.time()

    if len(faces) == 0:
        break

cap.release()