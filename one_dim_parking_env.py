import gym
from gym import spaces
import numpy as np

class OneDimParkingEnv(gym.Env):
    """
    One-Dimensional Parking Environment
    
    In this environment, an agent controls a car moving along a 1D line.
    The goal is to park the car (i.e., stop with zero velocity) inside a
    designated parking zone in the center of the track.
    
    State Space:
        - Position (x): Range from -5.0 to 5.0
        - Velocity (v): Range from -2.0 to 2.0
        
    Action Space:
        - 0: Accelerate backward (-1.0 m/s²)
        - 1: Coast (0.0 m/s²)
        - 2: Accelerate forward (+1.0 m/s²)
        
    Reward Function:
        - +1.0: Stopped in the center zone (i.e., parked successfully)
        - +0.5: In the parking zone but still moving
        - -0.1: Far away or moving past zone
        - -10.0: Crashed (out of bounds)
        
    Episode Termination:
        - Car is parked (stopped in the target zone)
        - Car crashes (goes outside the allowed area)
        - Maximum steps reached
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.dt = 1.0  # Time step for physics simulation
        self.x_range = (-5.0, 5.0)  # Position bounds
        self.v_range = (-2.0, 2.0)  # Velocity bounds
        self.target_zone = (-0.5, 0.5)  # Parking zone
        self.max_steps = 100  # Maximum steps per episode
        
        # Action mapping from discrete to continuous acceleration
        self.action_map = {
            0: -1.0,  # Accelerate backward
            1: 0.0,   # Coast
            2: 1.0    # Accelerate forward
        }
        
        # Define observation space: [position, velocity]
        self.observation_space = spaces.Box(
            low=np.array([self.x_range[0], self.v_range[0]]),
            high=np.array([self.x_range[1], self.v_range[1]]),
            dtype=np.float32
        )
        
        # Define action space: {0, 1, 2} = {backward, coast, forward}
        self.action_space = spaces.Discrete(3)
        
        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Reset the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for environment configuration
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        # Initialize the RNG
        super().reset(seed=seed)
        
        # Randomly initialize the car's position on the left side of the track
        self.x = self.np_random.uniform(-4.0, -1.0)
        self.v = 0.0  # Start with zero velocity
        self.steps = 0  # Reset step counter
        
        # Return the initial observation and info
        observation = np.array([self.x, self.v], dtype=np.float32)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take (0: backward, 1: coast, 2: forward)
            
        Returns:
            observation: New state after the action
            reward: Reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated (e.g., due to max steps)
            info: Additional information
        """
        # Map action to acceleration
        accel = self.action_map[action]
        
        # Update velocity (with clipping to respect bounds)
        self.v = np.clip(self.v + accel * self.dt, *self.v_range)
        
        # Update position (with clipping to respect bounds)
        self.x = np.clip(self.x + self.v * self.dt, *self.x_range)
        
        # Increment step counter
        self.steps += 1
        
        # Check if the episode is done
        terminated = False
        truncated = False
        
        # Calculate reward based on car state
        if self.target_zone[0] <= self.x <= self.target_zone[1]:
            if abs(self.v) < 1e-2:  # Car is practically stopped
                reward = 1.0  # Parked successfully
                terminated = True
            else:
                reward = 0.5  # In the zone but still moving
        elif self.x <= self.x_range[0] or self.x >= self.x_range[1]:
            reward = -10.0  # Crashed (out of bounds)
            terminated = True
        else:
            reward = -0.1  # Far away or moving past zone
        
        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            truncated = True
        
        # Create the observation and info dictionary
        observation = np.array([self.x, self.v], dtype=np.float32)
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment for human observation.
        """
        track_length = self.x_range[1] - self.x_range[0]
        position_normalized = (self.x - self.x_range[0]) / track_length
        target_start_normalized = (self.target_zone[0] - self.x_range[0]) / track_length
        target_end_normalized = (self.target_zone[1] - self.x_range[0]) / track_length
        
        # Create a text-based visualization
        track = "=" * 50
        target_start = int(50 * target_start_normalized)
        target_end = int(50 * target_end_normalized)
        car_pos = int(50 * position_normalized)
        
        # Construct the visualization
        visual = list(track)
        for i in range(target_start, target_end + 1):
            visual[i] = "P"  # P for parking
        
        if 0 <= car_pos < len(visual):
            visual[car_pos] = "C"  # C for car
        
        print("".join(visual))
        print(f"Position: {self.x:.2f}, Velocity: {self.v:.2f}")
        print(f"In target zone: {self.target_zone[0] <= self.x <= self.target_zone[1]}")
        print(f"Stopped: {abs(self.v) < 1e-2}")
        print("-" * 50)
    
    def close(self):
        """
        Clean up resources when environment is no longer needed.
        """
        pass
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            info: Dictionary with information
        """
        return {
            "distance_to_target_center": abs(self.x - (self.target_zone[0] + self.target_zone[1]) / 2),
            "in_target_zone": self.target_zone[0] <= self.x <= self.target_zone[1],
            "is_stopped": abs(self.v) < 1e-2
        }
