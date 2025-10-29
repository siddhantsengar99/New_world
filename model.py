import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers, optimizers
from keras.applications import MobileNetV2

def make_cnn(input_shape=(128, 128, 3), num_classes=43):
    # Load pretrained MobileNetV2 as base
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = True  # Freeze base initially to retain pretrained features

    # Build top layers
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    # Compile with a small learning rate
    model.compile(
        optimizer= optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
