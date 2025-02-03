import time
import os
import numpy as np
import cv2
import mss
import keyboard
import random
from collections import deque
from sklearn.model_selection import train_test_split

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

screenshot_counter = 0

# Game region for capturing the screen
region = {"top": 226, "left": 590, "width": 228, "height": 124}

def get_game_state():
    """Capture the game screen and preprocess it"""
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        image = np.array(screenshot)[:, :, :3]  # Remove alpha channel
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (200, 100)) / 255.0  # Normalize
        return resized_image.flatten()

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
    
    images = np.array(images).reshape(-1, 200 * 400) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels

print("Loading data...")
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded! Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Q-learning parameters
actions = [0, 1]  # 0: Jump, 1: Duck
state_size = 5000  # Limit state space size for hashing
q_table = np.zeros((state_size, len(actions)))
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

# Convert state into an index
def get_state_index(state):
    return hash(state.tobytes()) % state_size

# Choose action based on Q-table
def choose_action(state):
    state_idx = get_state_index(state)
    if np.random.rand() < epsilon:
        return np.random.choice(actions)  # Explore
    else:
        return np.argmax(q_table[state_idx])  # Exploit

# Update Q-table
def update_q_table(state, action, reward, next_state):
    state_idx = get_state_index(state)
    next_state_idx = get_state_index(next_state)
    best_next_action = np.argmax(q_table[next_state_idx])
    q_table[state_idx, action] = (1 - alpha) * q_table[state_idx, action] + alpha * (reward + gamma * q_table[next_state_idx, best_next_action])

# Train the Q-learning agent
def train_q_learning(episodes=1000):
    global epsilon
    for episode in range(episodes):
        state = get_game_state()
        done = False
        total_reward = 0
        
        while not done:
            time.sleep(0.1)  # Adjust delay as needed
            action = choose_action(state)
            
            if action == 0:
                keyboard.press_and_release("up")  # Jump
            elif action == 1:
                keyboard.press_and_release("down")  # Duck
            
            next_state = get_game_state()
            reward = 1  # Placeholder reward, modify based on game events
            done = False  # Modify based on game-over detection
            update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    np.save("q_table.npy", q_table)
    print("Training complete! Q-table saved as q_table.npy")

# Train the model
train_q_learning()
