#!/usr/bin/env python3
"""
Test script for the OneDimParkingEnv.
Runs a simple random agent to test functionality.
"""

import random
import sys
from one_dim_parking_env import OneDimParkingEnv

def run_random_episode(env, seed=None, render=False):
    """Run a single episode with random actions."""
    try:
        random.seed(seed)
        observation, info = env.reset(seed=seed)
        terminated = truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render and steps % 5 == 0:  # Render every 5 steps to reduce output
                print(f"\nStep {steps}, Action: {action}")
                env.render()
        
        success = info.get("is_parked", False)
        return {
            "steps": steps,
            "total_reward": total_reward,
            "success": success,
            "final_position": observation[0],
            "final_velocity": observation[1]
        }
    except Exception as e:
        print(f"Error during episode: {e}")
        return None

def main():
    """Run tests on the OneDimParkingEnv."""
    try:
        # Environment parameters test
        env = OneDimParkingEnv()
        print("\n== Environment Parameters ==")
        print(f"Target Zone: {env.target_zone}")
        
        # Run random episodes
        print("\n== Running 5 Random Episodes ==")
        
        results = []
        for i in range(5):
            print(f"\n--- Episode {i+1} ---")
            result = run_random_episode(env, seed=i, render=True)
            if result is not None:
                results.append(result)
                print(f"Episode ended after {result['steps']} steps with reward {result['total_reward']:.2f}")
                print(f"Final state: position={result['final_position']:.2f}, velocity={result['final_velocity']:.2f}")
                print(f"Success: {'Yes' if result['success'] else 'No'}")
        
        if results:
            # Summary statistics
            successes = sum(1 for r in results if r["success"])
            avg_reward = sum(r["total_reward"] for r in results) / len(results)
            avg_steps = sum(r["steps"] for r in results) / len(results)
            
            print("\n== Summary ==")
            print(f"Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Steps: {avg_steps:.1f}")
        else:
            print("No valid episodes completed. Check for errors above.")
        
        # Clean up
        env.close()
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
if __name__ == "__main__":
    main()