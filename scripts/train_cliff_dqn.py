from envs.grids import CliffWalking
from agents.dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import os

def train_cliff_walking():
    # Create environment
    env = CliffWalking()
    
    # Create agent
    agent = DQNAgent(
        env=env,
        state_dim=2,  # (x, y) coordinates
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train the agent
    save_dir = "exports"
    os.makedirs(save_dir, exist_ok=True)
    
    rewards = agent.train(
        episodes=1000,
        log_freq=10,
        save_path=save_dir # Save path for the trained model
    )
    
    # Plot learning curve
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('DQN Learning Curve on CliffWalking')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # Compute and plot moving average
    window_size = 50
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title(f'DQN Learning Curve (Moving Average, Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(f"{save_dir}/learning_curve.svg")
    plt.close()
    
    print(f"Training completed. Model saved to {save_dir}/dqn_cliff_final.pt")

if __name__ == "__main__":
    train_cliff_walking()