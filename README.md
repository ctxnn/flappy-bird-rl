# Flappy Bird RL with DQN

This project implements a Deep Q-Network (DQN) reinforcement learning agent to play Flappy Bird using PyTorch and Gymnasium.

## Features
- Basic DQN implementation
- Experience replay memory
- Configurable hyperparameters

## Setup
1. Clone this repository
2. Install dependencies:
   ```
   pip install torch gymnasium flappy-bird-gymnasium pyyaml
   ```
3. Run the agent:
   ```
   python agent.py
   ```

## TODO
- [ ] Implement Double DQN architecture
- [ ] Implement Dueling DQN architecture

## Hyperparameters
Configuration is managed through `hyperparameters.yml` file.