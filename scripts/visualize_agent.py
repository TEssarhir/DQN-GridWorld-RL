import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

from envs.grids import CliffWalking
from agents.dqn import DQNAgent

def visualize_episode(env, agent, max_steps=1000, save_path="exports"):
    """
    Run a single episode with the trained agent and visualize its path
    """
    state = env.reset()
    done = False
    states = [state]  # List to store states for visualization
    actions = []      # List to store actions
    steps = 0
    total_reward = 0
    
    # Run the episode
    while not done and steps < max_steps:
        action = agent.choose_action(state)
        actions.append(action)
        next_state, reward, done = env.step(action)
        state = next_state
        states.append(state)
        total_reward += reward
        steps += 1
    
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw the grid
    for i in range(env.width + 1):
        ax.axvline(i, color='black', linewidth=0.5)
    for i in range(env.height + 1):
        ax.axhline(i, color='black', linewidth=0.5)
    
    # Mark cliff positions
    for cliff_pos in env.falling_states:
        x, y = cliff_pos
        ax.add_patch(Rectangle((x, y), 1, 1, fill=True, color='red', alpha=0.5))
    
    # Mark start and goal
    ax.add_patch(Rectangle((env.start_state[0], env.start_state[1]), 1, 1, 
                          fill=True, color='green', alpha=0.5))
    ax.add_patch(Rectangle((env.goal_state[0], env.goal_state[1]), 1, 1, 
                          fill=True, color='blue', alpha=0.5))
    
    # Plot the agent's path
    path_x = [s[0] + 0.5 for s in states]
    path_y = [s[1] + 0.5 for s in states]
    ax.plot(path_x, path_y, 'o-', color='purple', markersize=8, linewidth=2)
    
    # Add action annotations
    for i, (x, y, a) in enumerate(zip(path_x[:-1], path_y[:-1], actions)):
        ax.annotate(a, (x, y), xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
    
    # Add step numbers
    for i, (x, y) in enumerate(zip(path_x, path_y)):
        ax.annotate(f"{i}", (x, y), xytext=(15, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=10)
    
    # Set limits and labels
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xticks(np.arange(0.5, env.width + 0.5))
    ax.set_yticks(np.arange(0.5, env.height + 0.5))
    ax.set_xticklabels(range(env.width))
    ax.set_yticklabels(range(env.height))
    ax.set_title(f'Agent Path in CliffWalking Environment\nTotal Reward: {total_reward}')
    
    # Add legend
    ax.add_patch(Rectangle((0, 0), 0, 0, fill=True, color='green', alpha=0.5, label='Start'))
    ax.add_patch(Rectangle((0, 0), 0, 0, fill=True, color='blue', alpha=0.5, label='Goal'))
    ax.add_patch(Rectangle((0, 0), 0, 0, fill=True, color='red', alpha=0.5, label='Cliff'))
    ax.plot([], [], 'o-', color='purple', label='Agent Path')
    ax.legend(loc='upper right')
    
    # Save the visualization
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/agent_path_visualization.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/agent_path_visualization.svg", bbox_inches='tight')
    plt.show()
    
    return states, actions, total_reward

def main():
    # Create environment
    env = CliffWalking()
    
    # Create agent
    agent = DQNAgent(
        env=env,
        state_dim=2,  # (x, y) coordinates
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        epsilon=0.01  # Low epsilon for exploitation
    )
    
    # Check if there's a saved model and load it
    save_dir = "exports"
    model_path = f"{save_dir}/dqn_agent_final.pth"
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        agent.load(model_path)
    else:
        print("No trained model found. Training a new agent...")
        agent.train(episodes=1000, save_path=save_dir)
    
    # Visualize the agent's movement
    visualize_episode(env, agent, save_path=save_dir)

if __name__ == "__main__":
    main()