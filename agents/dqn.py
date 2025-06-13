import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from agents.base import Agent
from envs.base import Environment
from models.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer

class DQNAgent(Agent):
    """Deep Q-Network agent"""
    
    def __init__(
        self, 
        env: Environment,
        state_dim: int,
        hidden_dim: int = 64,
        buffer_size: int = 10000,
        batch_size: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        update_target_freq: int = 100
    ):
        super().__init__(env)
        self.state_dim = state_dim
        self.action_dim = len(env.actions)
        self.actions = env.actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.train_step = 0
        
        # Initialize Q networks
        self.q_net = QNetwork(state_dim, self.action_dim, hidden_dim)
        self.target_q_net = QNetwork(state_dim, self.action_dim, hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def state_to_tensor(self, state):
        """Convert state to tensor for network input"""
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        with torch.no_grad():
            q_values = self.q_net(self.state_to_tensor(state))
            action_idx = torch.argmax(q_values).item()
            return self.actions[action_idx]
    
    def learn(self, state, action, reward, next_state, done):
        """Store experience in replay buffer and learn if enough samples"""
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Only learn if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions_idx = [self.actions.index(a) for a in actions]
        actions_idx = torch.tensor(actions_idx, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Compute current Q values
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes=500, max_steps=1000, log_freq=10, save_freq=100, save_path=None):
        """Train the agent for a specified number of episodes"""
        total_rewards = []
        
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            total_rewards.append(total_reward)
            
            if ep % log_freq == 0:
                avg_reward = sum(total_rewards[-log_freq:]) / min(log_freq, len(total_rewards))
                print(f"Episode {ep}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            if save_path and ep % save_freq == 0:
                self.save(f"{save_path}/dqn_agent_ep{ep}.pt")
        
        return total_rewards
    
    def save(self, path: str):
        """Save the agent to a file"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load the agent from a file"""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']