"""
Manual control for the OneDimParkingEnv environment.
This allows testing the environment using keyboard controls.
"""

from one_dim_parking_env import OneDimParkingEnv

def main():
    # Create and initialize the environment
    env = OneDimParkingEnv(render_mode="human")
    observation, info = env.reset()
    
    # Display controls to the user
    print("=" * 50)
    print("One-Dimensional Parking Environment - Manual Control")
    print("=" * 50)
    print("Controls:")
    print("  A: Accelerate backward (-1.0 m/s²)")
    print("  S: Coast (0.0 m/s²)")
    print("  D: Accelerate forward (+1.0 m/s²)")
    print("  Q: Quit")
    print("=" * 50)
    print("Goal: Park the car (stop) within the parking zone (P)")
    print("=" * 50)
    
    # Track episode stats
    total_reward = 0
    steps = 0
    terminated = False
    truncated = False
    
    # Main game loop
    while not (terminated or truncated):
        # Render the current state
        env.render()
        
        # Get user input
        valid_input = False
        while not valid_input:
            key = input("Action (A/S/D, Q to quit): ").lower()
            
            if key == 'a':
                action = 0  # Accelerate backward
                valid_input = True
            elif key == 's':
                action = 1  # Coast
                valid_input = True
            elif key == 'd':
                action = 2  # Accelerate forward
                valid_input = True
            elif key == 'q':
                print("Quitting...")
                return
            else:
                print("Invalid input. Use A, S, D, or Q.")
        
        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Update episode stats
        total_reward += reward
        steps += 1
        
        # Display step information
        print(f"Step: {steps}")
        print(f"Action: {['Backward', 'Coast', 'Forward'][action]}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        
    # End of episode
    env.render()
    print("=" * 50)
    print("Episode finished!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward}")
    
    if info["in_target_zone"] and info["is_stopped"]:
        print("Success! Car parked successfully in the target zone.")
    elif terminated:
        print("Car crashed or successfully parked.")
    elif truncated:
        print("Maximum number of steps reached.")
    
    env.close()

if __name__ == "__main__":
    main()
