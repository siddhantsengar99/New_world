import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMG_SIZE = 64  # square input

def load_dataset(root_dir, img_size=IMG_SIZE, classes=None, max_per_class=None):
    """Load images from root_dir where each subfolder is a class id (0,1,2,...)."""
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if classes:
        class_names = [c for c in class_names if c in classes]
    for cls in class_names:
        folder = os.path.join(root_dir, cls)
        files = os.listdir(folder)
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            path = os.path.join(folder, f)
            try:
                img = Image.open(path).convert('RGB').resize((img_size, img_size))
                arr = np.asarray(img)
                images.append(arr)
                labels.append(int(cls))
            except Exception as e:
                print("skip", path, e)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # normalize to [0,1]
    images = images / 255.0
    return images, labels

def get_train_val(data_dir, test_size=0.2, random_state=42):
    X, y = load_dataset(os.path.join(data_dir, 'train'))
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
