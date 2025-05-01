import cv2
import keyboard
import time

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Control frequency of shortcut trigger
last_trigger_time = 0
trigger_cooldown = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If exactly 2 faces are detected and cooldown passed
    if len(faces) == 2 and (time.time() - last_trigger_time > trigger_cooldown):
        keyboard.press_and_release('ctrl+shift+x')
        last_trigger_time = time.time()

    # Draw face boxes (optional)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
