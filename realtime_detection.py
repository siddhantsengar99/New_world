# realtime_demo.py
import time
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
from data_utils import IMG_SIZE
from labels import CLASS_LABELS  # your mapping dict: CLASS_LABELS[class_id] -> "Stop", etc.

# ---- load model ----
# ensure filename matches the model you saved
MODEL_PATH = 'traffic_sign_cnn.h5'
model = tf.keras.models.load_model(MODEL_PATH)
num_classes = model.output_shape[-1]

# ---- init text-to-speech once ----
tts = pyttsx3.init()
tts.setProperty('rate', 150)
tts.setProperty('volume', 1.0)

# ---- helper: preprocess + predict a crop (ROI) ----
def predict_frame(frame):
    """
    frame: BGR image (crop/ROI) from OpenCV
    returns: (class_id:int, confidence:float)
    """
    # convert BGR -> RGB, resize, normalize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)
    p = preds[0]
    class_id = int(np.argmax(p))
    confidence = float(np.max(p))
    return class_id, confidence

# ---- realtime camera setup ----
cap = cv2.VideoCapture(0)  # change index if you have multiple cameras
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

# voice-control variables to avoid spamming
last_spoken_label = None
last_spoken_time = 0.0
SPEECH_COOLDOWN = 4.0      # seconds between spoken alerts for same label
CONFIDENCE_THRESHOLD = 0.70  # only speak if confidence >= this

# main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # center-crop ROI (simple demo) — replace with detector for production
        min_dim = min(h, w)
        cx, cy = w // 2, h // 2
        half = min_dim // 3
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        roi = frame[y1:y2, x1:x2]

        text = "No ROI"
        label = None
        conf = 0.0

        if roi.size != 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            class_id, conf = predict_frame(roi)
            label = CLASS_LABELS.get(class_id, f"Class {class_id}")
            text = f"{label} ({conf*100:.1f}%)"

        # draw ROI rectangle + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Traffic Sign Recognition', frame)

        # voice alert logic: speak only when high confidence and cooldown passed
        now = time.time()
        if label is not None and conf >= CONFIDENCE_THRESHOLD:
            if label != last_spoken_label or (now - last_spoken_time) > SPEECH_COOLDOWN:
                # speak non-blocking: runAndWait is blocking, but short; acceptable for demo
                tts.say(f"{label} ahead")
                tts.runAndWait()
                last_spoken_label = label
                last_spoken_time = now

        # exit on ESC or 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
