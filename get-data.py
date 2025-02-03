import time
import os
import numpy as np
import cv2
import mss
import keyboard

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

screenshot_counter = 0

# Define the region to capture (adjust these values as needed)
region = {
    "top": 170,  # Y start
    "left": 525,  # X start
    "width": 1024,  # X range
    "height": 376  # Y range
}

def get_game_screenshot(action):
    """Capture a screenshot of the defined region and save in 'data' directory"""
    global screenshot_counter
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        image = np.array(screenshot)[:, :, :3]  # Remove alpha channel
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (400, 200))
        
        action_name = "arrow_up" if action == 0 else "arrow_down"
        filename = os.path.join("data", f"{action_name}_{screenshot_counter:04d}.png")
        cv2.imwrite(filename, resized_image)
        screenshot_counter += 1
        print(f"Saved: {filename}")

print("Press UP or DOWN to collect data. Press ESC to exit.")

while True:
    try:
        if keyboard.is_pressed("up"):
            get_game_screenshot(0)
            time.sleep(0.2)  # Prevent multiple captures per press
        elif keyboard.is_pressed("down"):
            get_game_screenshot(1)
            time.sleep(0.2)  # Prevent multiple captures per press
        elif keyboard.is_pressed("esc"):
            print("Exiting data collection...")
            break
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
