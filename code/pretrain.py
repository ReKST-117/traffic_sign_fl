import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# === 1. Set dataset path ===
data_path = "/home/nvidia/git/tsfl/dset"

# === 2. Load images and labels ===
images, labels = []
class_list = sorted(os.listdir(data_path))  # Ensure consistent class order

print("Detected classes:")
for i, cls in enumerate(class_list):
    print(f"{i}: {cls}")

for i, cls in enumerate(class_list):
    cls_path = os.path.join(data_path, cls)
    for fname in os.listdir(cls_path):
        img_path = os.path.join(cls_path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255.0
            images.append(img)
            labels.append(i)

images = np.array(images).reshape(-1, 32, 32, 1)
labels = to_categorical(labels, num_classes=len(class_list))

# === 3. Split dataset ===
X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# === 4. Build OtherNet feature extractor (without final output layer) ===
feature_extractor = Sequential([
    Conv2D(60, (5,5), activation='relu', input_shape=(32,32,1)),
    Conv2D(60, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(30, (3,3), activation='relu'),
    Conv2D(30, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5)
])

# === 5. Add temporary output layer for training ===
output = Dense(len(class_list), activation='softmax')(feature_extractor.output)
training_model = Model(inputs=feature_extractor.input, outputs=output)

# === 6. Compile and train the model ===
training_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

training_model.fit(X_train, Y_train,
                   epochs=10,
                   validation_data=(X_val, Y_val),
                   batch_size=32)

# === 7. Save the feature extractor only (without output layer) ===
feature_extractor.save("/home/nvidia/git/tsfl/feature_extractor_only.h5")

print("✅ Feature extractor model saved to /home/nvidia/git/tsfl/feature_extractor_only.h5")
