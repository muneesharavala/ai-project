import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, img_size=(224, 224)):
    """
    Loads and preprocesses all images from a folder.
    Converts them to grayscale, resizes, and normalizes.
    """
    images = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        if not os.path.isdir(label_path):
            continue  # Skip files that aren’t folders

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)

            # ✅ Skip non-image files (fix for .DS_Store and others)
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: could not read {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values (0–1)
                images.append(img)
                labels.append(label_folder)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

def split_data(X, y, test_size=0.2):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)
