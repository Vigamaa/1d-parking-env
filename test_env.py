"""
Test script for the OneDimParkingEnv environment.
This performs basic tests to validate the environment behavior.
"""

# Import the environment
from one_dim_parking_env import OneDimParkingEnv

def main():
    # Create the environment
    env = OneDimParkingEnv()
    
    # Print environment info
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test basic reset and step functionality
    print("\nTesting basic environment functionality:")
    
    # Reset the environment
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Take a random action
    action = env.action_space.sample()
    print(f"Taking random action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    
    print("\nEnvironment test complete!")

if __name__ == "__main__":
    main()
