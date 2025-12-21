import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from src.preprocessing import load_images_from_folder, split_data

# === Load data again for evaluation ===
data_path = "data/chest x-ray/train/"   # path to your X-ray dataset
images, labels = load_images_from_folder(data_path)
X_train, X_test, y_train, y_test = split_data(images, labels)

# Reshape for CNN
X_test = np.expand_dims(X_test, axis=-1)

# === Load trained model ===
model_path = "models/chest_xray_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# === Evaluate the model ===
test_loss, test_acc = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# === Predictions ===
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(tf.keras.utils.to_categorical(y_test), axis=1)

# === Print metrics ===
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

# === Confusion Matrix ===
cm = confusion_matrix(true_labels, pred_labels)
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.colorbar()
plt.show()
