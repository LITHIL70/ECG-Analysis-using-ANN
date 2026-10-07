import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft

model = load_model(os.path.join("model", "ann_model.h5"))
scaler = joblib.load(os.path.join("model", "scaler.joblib"))

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not supported")

    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    flat = img.flatten()

    fv = [
        np.mean(flat),
        np.std(flat),
        skew(flat),
        kurtosis(flat)
    ]
    fv.extend(np.abs(fft(flat))[:50])

    return np.array(fv).reshape(1, -1)

def predict_image(img_path):
    X = extract_features(img_path)
    X = scaler.transform(X)
    preds = model.predict(X)
    label = int(np.argmax(preds, axis=1)[0])

    classes = [
        "Normal",
        "Myocardial Infarction",
        "History of MI",
        "Abnormal Heartbeat"
    ]

    return classes[label]
