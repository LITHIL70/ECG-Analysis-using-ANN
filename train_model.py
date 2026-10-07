
import os
import csv
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from scipy.stats import skew, kurtosis
from scipy.fftpack import fft

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


base_path = os.path.join("datasets", "ecg data old version", "ecg data old version", "train")
non_ecg_path = os.path.join("datasets", "non_ecg")
model_dir = "model"
csv_path = "dataset.csv"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(non_ecg_path, exist_ok=True)

category_map = {
    "Normal Person ECG Images (284x12=3408)": 0,
    "ECG Images of Myocardial Infarction Patients (240x12=2880)": 1,
    "ECG Images of Patient that have History of MI (172x12=2064)": 2,
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 3
}

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])

    for folder, label in category_map.items():
        full_folder = os.path.join(base_path, folder)
        if not os.path.exists(full_folder):
            continue
        for img in os.listdir(full_folder):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                writer.writerow([os.path.join(full_folder, img), label])

    for img in os.listdir(non_ecg_path):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            writer.writerow([os.path.join(non_ecg_path, img), 4])

df = pd.read_csv(csv_path)

features = []
labels = []
img_size = 64

for _, row in df.iterrows():
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    flat = img.flatten()

    if np.std(flat) < 1e-6:
        continue

    mean_val = np.mean(flat)
    std_val = np.std(flat)
    skew_val = skew(flat)
    kurt_val = kurtosis(flat)

    if not np.isfinite(skew_val) or not np.isfinite(kurt_val):
        continue

    fft_vals = np.abs(fft(flat))[:50]

    fv = [mean_val, std_val, skew_val, kurt_val]
    fv.extend(fft_vals)

    features.append(fv)
    labels.append(row["label"])

X = np.array(features, dtype=np.float32)
y = np.array(labels)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

model = Sequential([
    Dense(256, activation="relu", input_dim=X.shape[1]),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train_cat,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Myocardial Infarction",
            "History of MI",
            "Abnormal Heartbeat",
            "Non-ECG"
        ],
        zero_division=0
    )
)


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "MI", "History MI", "Abnormal", "Non-ECG"],
    yticklabels=["Normal", "MI", "History MI", "Abnormal", "Non-ECG"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value (%)": [
        round(accuracy * 100, 2),
        round(precision * 100, 2),
        round(recall * 100, 2),
        round(f1 * 100, 2)
    ]
})

print("\nOverall Performance Metrics")
print(metrics_df)

metrics_df.to_csv("performance_metrics.csv", index=False)

model.save(os.path.join(model_dir, "ann_model.h5"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

