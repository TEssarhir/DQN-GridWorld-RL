from envs.grids import CliffWalking
from agents.dqn import DQNAgent

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
        epsilon=0.1
    )
    
    # Train the agent
    agent.train(episodes=1000)
    
    # Save the trained agent
    agent.save("dqn_agent.pt")

if __name__ == "__main__":
    main()