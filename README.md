# DQN GridWorld Reinforcement Learning

A Python implementation of Deep Q-Networks (DQN) for solving grid world reinforcement learning problems, specifically the CliffWalking environment.

## Project Overview

This project demonstrates the application of Deep Q-Learning to solve the classic CliffWalking problem in reinforcement learning. The agent learns to navigate from a starting position to a goal while avoiding dangerous "cliff" cells in the grid environment.

## Features

- Implementation of Deep Q-Network (DQN) with experience replay
- Custom grid world environments with configurable parameters
- Visualization tools for agent trajectories
- Training scripts with learning curve visualization
- Modular and extensible architecture

## Project Structure

```txt
DQN-GridWorld-RL/
├── agents/               # Agent implementations
│   ├── base.py           # Base abstract agent class
│   └── dqn.py            # DQN agent implementation
├── envs/                 # Environment implementations
│   ├── base.py           # Base environment class
│   └── grids.py          # Grid world environments (CliffWalking)
├── models/               # Neural network models
│   └── q_network.py      # Q-Network architecture
├── scripts/              # Utility scripts
│   ├── train_cliff_dqn.py    # Script for training DQN on CliffWalking
│   └── visualize_agent.py    # Script for visualizing agent behavior
├── utils/                # Utility functions and classes
│   └── replay_buffer.py  # Experience replay buffer
├── main.py               # Main entry point
└── requirements.txt      # Project dependencies
```

## Installation

```bash
git clone https://github.com/TEssarhir/DQN-GridWorld-RL.git
cd DQN-GridWorld-RL
```

## Usage

### 1. Training a DQN Agent

To train a DQN agent on the CliffWalking environment:

```bash
python scripts/train_cliff_dqn.py
```

This will :

- Create a CliffWalking environment
- Initialize a DQN agent
- Train the agent for 1000 episodes
- Save the trained model to the "exports" directory
- Generate a learning curve visualization

### 2. Visualizing Agent Behavior

To visualize how a trained agent performs

```bash
python scripts/visualize_agent.py
```

This will:

Load a trained model (or train one if not found)
Run the agent in the environment
Create a visualization of the agent's path
Save the visualization to the "exports" directory

## Implementation Details

### 1. Environment

The CliffWalking environment is a grid world where:

- The agent starts at position (0,0)
- The goal is at the opposite corner
- Certain cells are "cliffs" which cause the agent to fall and receive a negative reward
- The agent can move in four directions: up, down, left, right
- Rewards: -1 for each step, -100 for falling off a cliff, +100 for reaching the goal

### 2. DQN Agent

The DQN implementation includes:

- A neural network for approximating Q-values
- Experience replay buffer for storing and sampling transitions
- Target network for stable learning
- Epsilon-greedy exploration strategy
- Periodic model saving

### 3. Customization

You can customize various aspects of the implementation:

- Grid dimensions and cliff positions in CliffWalking class
- Neural network architecture in QNetwork class
- DQN hyperparameters like learning rate, discount factor, and exploration rate
