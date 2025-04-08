import cv2
import os
import time
import requests
import pyttsx3
from twilio.rest import Client
from ultralytics import YOLO
from deepface import DeepFace

TWILIO_PHONE = "+14128735632"
RECIPIENT_PHONES = ["+917632910105"]
TWILIO_SID = os.getenv("AC39afa99557af7551f57e8a8206465a6e")
TWILIO_AUTH_TOKEN = os.getenv("91b9bcf21b01e75f747b2d22d0862a3b")

TELEGRAM_BOT_TOKEN = "7870072374:AAEkNThqVJZOLSAQdKFMGwweggL2hm5enaM"
TELEGRAM_CHAT_ID = "859772480"

model = YOLO('yolov8n.pt')

FACE_DATASET = r"C:\Users\BIT\Desktop\Python\known_faces/"
if not os.path.exists(FACE_DATASET):
    os.makedirs(FACE_DATASET)

cap = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 0.5
alert_sent = False

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
engine = pyttsx3.init()

def send_telegram_alert(person_name):
    try:
        message = f"Alert! {person_name} has been detected in the monitored area."
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = requests.post(url, params=params)
        if response.status_code == 200:
            print(f"Telegram alert sent: {person_name}")
        else:
            print("Failed to send Telegram alert:", response.text)
    except Exception as e:
        print("Error in sending Telegram message:", e)

def make_call():
    global alert_sent
    if alert_sent:
        return
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        for phone in RECIPIENT_PHONES:
            call = client.calls.create(
                twiml='<Response><Say>Alert! An unknown person has been detected.</Say></Response>',
                from_=TWILIO_PHONE,
                to=phone
            )
            print(f"Call alert sent to {phone}!")
        alert_sent = True
        time.sleep(60)
        alert_sent = False
    except Exception as e:
        print("Call failed:", e)

def recognize_face(face_img):
    try:
        for person in os.listdir(FACE_DATASET):
            person_folder = os.path.join(FACE_DATASET, person)
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                result = DeepFace.verify(face_img, img_path, model_name='VGG-Face', enforce_detection=False)
                if result['verified']:
                    return person
        return "Unknown"
    except Exception as e:
        print("Face recognition error:", e)
        return "Error"

def greet_person(person_name):
    greeting_text = f"Hello, {person_name}!"
    engine.say(greeting_text)
    engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detected_person = None

    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        for box in results[0].boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model.names[class_id]

            if label == "person" and confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (fx, fy, fw, fh) in faces:
        face_img = frame[fy:fy + fh, fx:fx + fw]
        person_name = recognize_face(face_img)

        if person_name != "Unknown":
            detected_person = person_name
            greet_person(person_name)

        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(frame, person_name, (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if detected_person:
        send_telegram_alert(detected_person)
    else:
        send_telegram_alert("Unknown Person")
        make_call()

    cv2.imshow('Human & Face Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
