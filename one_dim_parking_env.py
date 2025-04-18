#!/usr/bin/env python3
"""
One-Dimensional Parking Environment.

A simple environment for reinforcement learning where an agent 
controls a car that needs to park in a designated zone.
"""

import random
import math


class OneDimParkingEnv:
    """
    One-Dimensional Parking Environment
    
    An environment where an agent controls a car moving in one dimension.
    The goal is to stop exactly inside a designated parking zone.
    
    State:
        Position (x): position on a line from -5 to 5
        Velocity (v): speed of the car from -2 to 2
        
    Action:
        0: Accelerate backward (-1.0 m/s²)
        1: Coast (0.0 m/s²)
        2: Accelerate forward (1.0 m/s²)
        
    Reward:
        +1.0: Stopped in center zone
        +0.5: In zone but moving
        -0.1: Far away or moving past zone
        -10.0: Crashed (outside bounds)
        
    Episode Termination:
        - Car successfully parked (stopped in target zone)
        - Car crashed (went outside track bounds)
        - Max steps reached
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        # Physics parameters
        self.dt = 1.0  # Time step in seconds
        self.x_range = (-5.0, 5.0)  # Track limits
        self.v_range = (-2.0, 2.0)  # Velocity limits
        self.target_zone = (-0.5, 0.5)  # Parking zone
        self.max_steps = 100  # Maximum episode length
        
        # Action mapping: index → acceleration
        self.action_map = {
            0: -1.0,  # Accelerate backward
            1: 0.0,   # Coast
            2: 1.0    # Accelerate forward
        }
        
        # Define action space: 3 discrete actions
        self.action_space = ActionSpace(3)
        
        # Rendering setup
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.render_mode = render_mode
        
        # State variables
        self.x = 0.0
        self.v = 0.0
        self.steps = 0
        self.rng = Random()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used)
            
        Returns:
            observation: Initial state [position, velocity]
            info: Additional info
        """
        # Seed the RNG if provided
        if seed is not None:
            self.rng.seed(seed)
        
        # Initialize state: random position, zero velocity
        self.x = self.rng.uniform(-4.0, -1.0)  # Start left of target
        self.v = 0.0
        self.steps = 0
        
        # Return observation and info
        observation = [self.x, self.v]
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action (int): Action to take (0: backward, 1: coast, 2: forward)
            
        Returns:
            observation: New state [position, velocity]
            reward: Reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated (e.g., time limit)
            info: Additional info
        """
        # Get acceleration from action
        accel = self.action_map[action]
        
        # Update velocity and position using simple physics
        self.v = clip(self.v + accel * self.dt, self.v_range[0], self.v_range[1])
        self.x = clip(self.x + self.v * self.dt, self.x_range[0], self.x_range[1])
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Calculate reward
        if self.target_zone[0] <= self.x <= self.target_zone[1]:
            # In target zone
            if abs(self.v) < 1e-2:  # Effectively stopped
                reward = 1.0
                terminated = True  # Successfully parked
            else:
                reward = 0.5  # In zone but still moving
        elif self.x <= self.x_range[0] or self.x >= self.x_range[1]:
            # Crashed into boundary
            reward = -10.0
            terminated = True
        else:
            # Not in target zone
            reward = -0.1
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            truncated = True
        
        # Build observation and info
        observation = [self.x, self.v]
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            A rendered frame depending on the render_mode
        """
        if self.render_mode is None:
            return None
            
        # Create the visual representation
        track = "-" * 50
        pos_idx = int((self.x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 49)
        target_start = int((self.target_zone[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 49)
        target_end = int((self.target_zone[1] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 49)
        
        # Create the visual representation
        visual = list(track)
        # Add target zone
        for i in range(target_start, target_end + 1):
            visual[i] = "P"
        # Add car position
        visual[pos_idx] = "C"
        
        visual_str = "".join(visual)
        status = f"Position: {self.x:.2f}, Velocity: {self.v:.2f}"
        
        if self.target_zone[0] <= self.x <= self.target_zone[1] and abs(self.v) < 1e-2:
            status += "\nSuccessfully parked!"
            
        if self.render_mode == "human":
            # Print to console
            print(visual_str)
            print(status)
            return None
            
        elif self.render_mode == "ansi":
            return visual_str + "\n" + status
    
    def _get_info(self):
        """
        Return additional information about the environment state.
        """
        return {
            "distance_to_target": abs((self.target_zone[0] + self.target_zone[1])/2 - self.x),
            "velocity": self.v,
            "in_target_zone": self.target_zone[0] <= self.x <= self.target_zone[1],
            "is_parked": self.target_zone[0] <= self.x <= self.target_zone[1] and abs(self.v) < 1e-2
        }

    def close(self):
        """
        Clean up resources when environment is no longer needed.
        """
        pass


class ActionSpace:
    """Simple discrete action space."""
    
    def __init__(self, n):
        """Initialize with n possible actions from 0 to n-1."""
        self.n = n
        
    def sample(self):
        """Sample a random action."""
        return random.randint(0, self.n - 1)


class Random:
    """Simple random number generator wrapper."""
    
    def __init__(self, seed=None):
        """Initialize with optional seed."""
        self.seed(seed)
        
    def seed(self, seed=None):
        """Set the random seed."""
        random.seed(seed)
        return [seed]
        
    def uniform(self, low, high):
        """Generate a uniform random number between low and high."""
        return random.uniform(low, high)


def clip(value, min_value, max_value):
    """Clip a value between min and max."""
    return max(min_value, min(value, max_value))