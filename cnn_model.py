from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, concatenate
)
from tensorflow.keras.models import Model
import tensorflow as tf

# -------------------------------
# U-NET ARCHITECTURE
# -------------------------------
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation="relu", padding="same")(p2)

    # Decoder
    u1 = UpSampling2D()(c3)
    m1 = concatenate([u1, c2])
    c4 = Conv2D(64, 3, activation="relu", padding="same")(m1)

    u2 = UpSampling2D()(c4)
    m2 = concatenate([u2, c1])
    c5 = Conv2D(32, 3, activation="relu", padding="same")(m2)

    outputs = Conv2D(1, 1, activation="sigmoid")(c5)

    return Model(inputs, outputs)

# -------------------------------
# METRICS
# -------------------------------
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
