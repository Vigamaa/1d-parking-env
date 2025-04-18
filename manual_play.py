#!/usr/bin/env python3
"""
Manual control interface for the 1D Parking Environment.
Use keyboard controls to test the environment:
- A: Accelerate backward
- S: Coast (no acceleration)
- D: Accelerate forward
- Q: Quit
"""

import sys
from one_dim_parking_env import OneDimParkingEnv

def main():
    """Run manual control loop for the 1D Parking Environment."""
    # Create environment
    env = OneDimParkingEnv(render_mode="human")
    observation, info = env.reset()
    
    # Print initial state
    print("\n=========== 1D Parking Environment ===========")
    print("Goal: Park the car (C) in the parking zone (P)")
    print("Controls:")
    print("  A: Accelerate backward")
    print("  S: Coast (no acceleration)")
    print("  D: Accelerate forward")
    print("  Q: Quit")
    print("=============================================\n")
    
    env.render()
    
    # Main control loop
    total_reward = 0
    step = 0
    terminated = truncated = False
    
    while not (terminated or truncated):
        # Get user action
        valid_action = False
        action = None
        
        while not valid_action:
            user_input = input("\nAction (A/S/D, Q to quit): ").lower()
            
            if user_input == 'a':
                action = 0  # Backward
                valid_action = True
            elif user_input == 's':
                action = 1  # Coast
                valid_action = True
            elif user_input == 'd':
                action = 2  # Forward
                valid_action = True
            elif user_input == 'q':
                print("Quitting...")
                env.close()
                return
            else:
                print("Invalid input. Use A, S, D, or Q.")
        
        # Action mapping for display
        action_mapping = {0: "Backward", 1: "Coast", 2: "Forward"}
        
        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Update and display information
        total_reward += reward
        step += 1
        
        print(f"\nStep {step}:")
        env.render()
        print(f"Action: {action_mapping.get(action, 'Unknown')}")
        print(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(f"Info: {info}")
        
    # Episode end
    print("\n=========== Episode Complete ===========")
    if info.get("is_parked", False):
        print("Success! Car parked correctly.")
    elif observation[0] <= env.x_range[0] or observation[0] >= env.x_range[1]:
        print("Crashed! Car went out of bounds.")
    else:
        print("Time limit reached without parking.")
    
    print(f"Total steps: {step}")
    print(f"Final reward: {total_reward:.2f}")
    
    # Clean up
    env.close()
    
if __name__ == "__main__":
    main()