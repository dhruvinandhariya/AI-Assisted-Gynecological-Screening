import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from cnn_model import build_unet

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "data/abdominal_US/AUS/images/train"
MASK_DIR  = "data/abdominal_US/AUS/annotations/train"

IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 8

MODEL_OUT = "models/ultrasound_unet.h5"
os.makedirs("models", exist_ok=True)

# -----------------------------
# METRICS (ADDED)
# -----------------------------
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def iou_score(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / (union + 1e-7)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(img_dir, mask_dir):
    X, y = [], []

    image_files = sorted(os.listdir(img_dir))

    for file in image_files:
        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(mask_dir, file)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        y.append(mask)

    return np.array(X), np.array(y)

print("[INFO] Loading training data...")
X_train, y_train = load_data(IMAGE_DIR, MASK_DIR)

print("[INFO] X_train shape:", X_train.shape)
print("[INFO] y_train shape:", y_train.shape)

if len(X_train) == 0:
    raise ValueError("No training data found. Check dataset paths.")

# -----------------------------
# BUILD MODEL
# -----------------------------
model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", dice_coefficient, iou_score]  # 🔹 UPDATED
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
print("[INFO] Starting training...")
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(MODEL_OUT)
print(f"[INFO] Model saved to {MODEL_OUT}")
