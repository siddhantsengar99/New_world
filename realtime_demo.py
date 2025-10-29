# realtime_demo.py
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
from model import make_cnn
from data_utils import IMG_SIZE
from labels import LABELS

# load model
model = tf.keras.models.load_model('traffic_sign_cnn.h5')
num_classes = model.output_shape[-1]

# init tts
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# helper predict function
def predict_frame(frame):
    # frame: BGR image from OpenCV
    # Preprocess: convert to RGB, resize to IMG_SIZE
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    p = preds[0]
    class_id = int(np.argmax(p))
    confidence = float(np.max(p))
    return class_id, confidence

# main
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

alerted = {}  # to avoid repeating alerts too rapidly

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    # For demo simplicity use center crop as ROI. In production use detector + classifier.
    min_dim = min(h, w)
    cx, cy = w // 2, h // 2
    half = min_dim // 3  # ROI size
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    roi = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

    if roi.size != 0:
        class_id, conf = predict_frame(roi)
        label = LABELS.get(class_id, f"Class {class_id}")
        text = f"{label} ({conf*100:.1f}%)"
    else:
        text = "No ROI"

    # draw
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Traffic Sign Recognition', frame)

    # voice alert logic: e.g., STOP or speed limit
    # change these keys according to labels you care about
    alert_keys = ["Stop", "Speed limit (30km/h)", "Speed limit (50km/h)"]
    for ak in alert_keys:
        if ak.lower() in label.lower() and conf > 0.75:
            last = alerted.get(ak, 0)
            import time
            if time.time() - last > 4.0:  # 4s cooldown
                tts.say(ak)
                tts.runAndWait()
                alerted[ak] = time.time()

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
