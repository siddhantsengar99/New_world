
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import get_train_val, load_dataset, IMG_SIZE
from model import make_cnn



print("Current working directory:", os.getcwd())
print("Expected dataset path:", os.path.join('dataset', 'train'))
print("Path exists:", os.path.exists(os.path.join('dataset', 'train')))



DATA_DIR = 'dataset'  # change if needed
BATCH_SIZE = 64
EPOCHS = 25

# load data
X_train, X_val, y_train, y_val = get_train_val(DATA_DIR, test_size=0.15)
num_classes = len(np.unique(np.concatenate([y_train, y_val])))

# data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    brightness_range=(0.7, 1.3)
)
train_gen = train_datagen.flow(X_train, tf.keras.utils.to_categorical(y_train, num_classes), batch_size=BATCH_SIZE)

val_datagen = ImageDataGenerator()
val_gen = val_datagen.flow(X_val, tf.keras.utils.to_categorical(y_val, num_classes), batch_size=BATCH_SIZE, shuffle=False)

# model
model = make_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(train_gen,
                    epochs=EPOCHS,
                    validation_data=val_gen,
                    callbacks=[checkpoint, reduce_lr],
                    steps_per_epoch=len(train_gen))

# load best
model.load_weights('best_model.h5')

# evaluate
val_preds = model.predict(X_val)
y_pred = np.argmax(val_preds, axis=1)
print(classification_report(y_val, y_pred))

# save final
model.save('traffic_sign_cnn.h5')

# plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')

plt.show()
