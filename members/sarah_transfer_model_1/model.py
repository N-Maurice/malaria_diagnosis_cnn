"""Transfer Learning Model 1 (EfficientNetB0) — owned by Sarah."""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = 224

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def build_model(fine_tune_at=None, dropout_rate=0.3):
    """Builds EfficientNetB0 with a binary classification head.

    fine_tune_at: number of layers to keep frozen from the start of the base
    model; layers after this index are trainable. None keeps the whole base frozen.
    """
    base = EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    if fine_tune_at is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs), base


# Final selected configuration (experiment 06): last 40 layers of EfficientNetB0
# unfrozen, dropout 0.3, Adam optimizer at learning rate 1e-5.
