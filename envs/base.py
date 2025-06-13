from abc import ABC, abstractmethod
from typing import Tuple, Any, Set, List

class Environment(ABC):
    """Base class for all environments"""
    
    @property
    @abstractmethod
    def actions(self):
        """Return the set of available actions"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial state"""
        pass
    
    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool]:
        """Take a step in the environment
        
        Returns:
            state: The new state
            reward: The reward for this step
            done: Whether the episode is done
        """
        pass