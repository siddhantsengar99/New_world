import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import utils, callbacks, optimizers
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from data_utils import get_train_val, load_dataset, IMG_SIZE
from model import make_cnn

# Basic setup
DATA_DIR = 'dataset'
BATCH_SIZE = 64
EPOCHS = 25
print("Starting training process...")

# Load dataset and split into training and validation sets
X_train, X_val, y_train, y_val = get_train_val(DATA_DIR, test_size=0.15)
num_classes = len(np.unique(np.concatenate([y_train, y_val])))

# Data augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True
)

# Validation data is only normalized, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(
    X_train, utils.to_categorical(y_train, num_classes),
    batch_size=BATCH_SIZE
)

val_gen = val_datagen.flow(
    X_val, utils.to_categorical(y_val, num_classes),
    batch_size=BATCH_SIZE, shuffle=False
)

# Build and compile CNN model
model = make_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks help stabilize and improve training performance
checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
lr_schedule = callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 0.95**epoch)

# Train the model with validation
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, reduce_lr, early_stop, lr_schedule],
    steps_per_epoch=len(train_gen)
)

# Load the best weights after training
model.load_weights('best_model.h5')

# Evaluate on validation set
val_preds = model.predict(X_val / 255.0)
y_pred = np.argmax(val_preds, axis=1)
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

# Save final trained model
model.save('traffic_sign_cnn.h5')
print("Training complete. Model saved as traffic_sign_cnn.h5")

# Plot training and validation performance
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy Curve')

plt.show()
