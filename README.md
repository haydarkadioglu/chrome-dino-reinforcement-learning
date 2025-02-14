# Chrome Dino Reinforcement Learning

A machine learning project that teaches an AI agent to play the Chrome Dinosaur game using Deep Q-Learning (DQN) and computer vision.

## Overview

This project consists of two main parts:
1. Data collection from gameplay
2. Training a DQN model to play the game

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- mss (screen capture)
- keyboard
- scikit-learn

## Project Structure

- `get-data.py` - Script to collect training data by capturing screen regions during gameplay
- `train.py` - Main training script that implements the DQN model
- `train.ipynb` - Jupyter notebook version of the training process with Q-learning visualization
- `data/` - Directory containing captured gameplay images
- `dino_dqn_model.h5` - Trained model weights

## How It Works

1. **Data Collection**:
   - Run `get-data.py`
   - Play the Chrome Dino game using UP/DOWN arrow keys
   - The script captures screen regions and saves them as images
   - Images are labeled based on the action (jump/duck)

2. **Training**:
   - Images are preprocessed (grayscale, resized to 400x200)
   - A CNN-based DQN model processes the game state
   - Model learns to predict optimal actions (jump/duck)
   - Training uses Q-learning with experience replay

## Model Architecture
The DQN model consists of:

- 2 Convolutional layers
- 2 MaxPooling layers
- Dense layers
- Output: 2 actions (jump/duck)

## Usage

1. Data Collection:
```bash
python get-data.py
```


# Q-Learning Algorithm for Game Simulation

This script implements a basic Q-learning algorithm for a game simulation where an agent needs to learn whether to "Jump" or "Duck" based on its state. The objective is to train the agent to maximize its rewards by updating the Q-table.

## Parameters

- **actions**: List of possible actions the agent can take. In this case:
  - `0`: Jump
  - `1`: Duck
  
- **state_size**: The size of the state space. This is determined by the shape of `X_train`, where `X_train.shape[1]` and `X_train.shape[2]` represent the dimensions of the state space.
  
- **q_table**: A table that stores the Q-values for each state-action pair. Initially, all Q-values are set to 0.

- **alpha (Learning Rate)**: A constant that determines how much new information is incorporated into the Q-values. Default is `0.1`.

- **gamma (Discount Factor)**: A constant that determines the importance of future rewards. Default is `0.9`.

- **epsilon (Exploration-Exploitation Tradeoff)**: A constant that controls the exploration-exploitation balance. At the start, the agent explores more (epsilon = 1.0), and over time, it gradually shifts towards exploitation (with epsilon decay). Default is `1.0`.

## Functions

### `choose_action(state)`
This function chooses an action based on the current state. It uses an epsilon-greedy strategy:
- With probability `epsilon`, the agent explores by choosing a random action.
- With probability `1 - epsilon`, the agent exploits its knowledge by choosing the action with the highest Q-value in the current state.

### `update_q_table(state, action, reward, next_state)`
This function updates the Q-table using the Q-learning update rule:
- It calculates the best possible action in the next state using the Q-values.
- It updates the Q-value for the current state-action pair by considering the reward and the discounted Q-value of the next state.

## Training Loop

The agent is trained over a number of episodes (`num_episodes`), and during each episode:
1. A random initial state is chosen.
2. The agent takes actions until the game ends or reaches a certain condition (simulated here with a 10% chance of "game over").
3. For each action taken, the Q-table is updated based on the reward received.

During training:
- **epsilon** decays over time to shift from exploration to exploitation. The decay rate is controlled by `epsilon_decay`, and the minimum value for epsilon is `min_epsilon`.

## Key Parameters

- **num_episodes**: The number of episodes the agent will train for. Default is `1000`.
- **epsilon_decay**: The rate at which epsilon decays after each episode. Default is `0.995`.
- **min_epsilon**: The minimum value epsilon can reach. Default is `0.01`.

## Example Output

During each episode, the total reward for that episode is printed. After training, the Q-learning process completes, and the agent will have learned the best actions to take in various states based on the training.

## Notes

- The reward in this example is a dummy value (`1`). In a real game, the reward should be calculated based on the outcome of the action (e.g., whether jumping or ducking was the correct decision).
- This code assumes that `X_train` is available, but it doesn't provide the data or context for how `X_train` is generated. It should represent the state space for the game or simulation.
  
## Conclusion

This Q-learning implementation allows the agent to learn optimal actions in a game by exploring and exploiting its environment. The model's performance improves as the agent trains over more episodes and fine-tunes its action-value estimates.


