from envs.base import Environment

class CliffWalking(Environment):
    """
    CliffWalking Environment for Reinforcement Learning
    ========
    A grid-world implementation of the classic Cliff Walking problem, commonly used in 
    reinforcement learning research and education. The agent navigates from a start state 
    to a goal state while avoiding falling off cliffs.
    Features:
    - Customizable grid dimensions (default: 8x8)
    - Four directional movement actions: up, down, left, right
    - Configurable "cliff" positions that reset the agent with negative reward
    - Goal-reaching detection with positive reward
    - Boundary detection to prevent the agent from leaving the grid
    Rewards:
    - Default step reward: -1 (small penalty for each action)
    - Falling off cliff: -100 (large penalty)
    - Reaching goal: +100 (large reward)
    Example Usage:
        env = CliffWalking()
        env.reset()
        state, reward, done = env.step('right')
    """
    
    def __init__(
            self,
            width=8, height=8,
            falling_states=[(2,2), (2,3), (2,4), (2,5)],
        ):
        # Define available moves but NOT actions as instance attribute
        self._actions = ['up', 'down', 'left', 'right']  # Changed to _actions as private attribute
        self.moves = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        self.width = width
        self.height = height
        self.start_state = (0, 0)
        self.goal_state = (width - 1, height - 1)
        self.state = self.start_state
        self.falling_states = falling_states
        self.rewards = -1  # Default reward for each step

    @property
    def actions(self):
        """Return the available actions"""
        return self._actions  # Return the private _actions attribute

    def reset(self):
        """Reset the environment to the initial state."""
        self.state = self.start_state
        return self.state

    def in_grid(self) -> bool:
        """Check if the current state is within the grid bounds."""
        x, y = self.state
        return 0 <= x < self.width and 0 <= y < self.height
    
    def step(self, action) -> tuple:
        """Take a step in the environment based on the action."""
        x, y = self.state

        # Apply the movement
        dx, dy = self.moves.get(action, (0, 0))
        self.state = (x + dx, y + dy)

        # Check if the agent has reached the goal
        if self.state == self.goal_state:
            return self.state, 100, True
        
        # Check if the agent has fallen off the cliff
        elif self.state in self.falling_states or not self.in_grid():
            self.reset()
            return self.state, -100, True
        
        else:
            return self.state, self.rewards, False