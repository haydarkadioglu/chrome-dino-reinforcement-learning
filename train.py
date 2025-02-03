import time
import os
import numpy as np
import cv2
import mss
import tensorflow as tf
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

screenshot_counter = 0

# Load collected data
def load_data():
    """Load collected screenshots and labels into numpy arrays"""
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (400, 200))
        images.append(image)
        
        # Label encoding: arrow_up -> 0, arrow_down -> 1
        label = 0 if "arrow_up" in filename else 1
        labels.append(label)
    
    images = np.array(images).reshape(-1, 200, 400, 1) / 255.0  # Normalize
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=2)  # Convert to one-hot encoding
    return images, labels

print("Loading data...")
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded! Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Define Deep Q-Network (DQN) model
def build_dqn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 400, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Two actions: jump or duck
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize DQN
model = build_dqn_model()

# Train the DQN model
def train_dqn(epochs=10, batch_size=1):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    model.save("dino_dqn_model.h5")
    print("Training complete! Model saved as dino_dqn_model.h5")

# Train the model
train_dqn()
