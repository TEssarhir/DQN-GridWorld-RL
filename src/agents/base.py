from abc import ABC, abstractmethod
from envs.base import Environment

class Agent(ABC):
    """Base class for all agents"""
    
    def __init__(self, env: Environment):
        self.env = env
    
    @abstractmethod
    def choose_action(self, state):
        """Choose an action based on the current state"""
        pass
    
    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """Learn from experience"""
        pass
    
    @abstractmethod
    def train(self, episodes: int):
        """Train the agent for a number of episodes"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the agent to a file"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the agent from a file"""
        pass