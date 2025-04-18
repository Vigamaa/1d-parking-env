#!/usr/bin/env python3
"""
Reinforcement Learning agent for 1D Parking Environment.

This file implements a simple Q-learning agent that learns
to park a car in a 1D environment.
"""

import random
import math
import time
from one_dim_parking_env import OneDimParkingEnv


class QLearningAgent:
    """Q-learning agent for the 1D Parking Environment."""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.01):
        """Initialize the Q-learning agent.
        
        Args:
            env: The environment to interact with
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for Q-learning
            exploration_rate: Initial epsilon for epsilon-greedy exploration
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        # Since we have a continuous state space, we'll discretize it
        self.position_bins = 20  # Number of bins for position
        self.velocity_bins = 10  # Number of bins for velocity
        
        # Create the Q-table with zeros
        self.q_table = {}  # We'll use a dict for sparse representation
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table lookup."""
        # Discretize position: [-5.0, 5.0] -> [0, position_bins-1]
        position = state[0]
        pos_bin = int((position - self.env.x_range[0]) / 
                      (self.env.x_range[1] - self.env.x_range[0]) * self.position_bins)
        pos_bin = max(0, min(self.position_bins - 1, pos_bin))
        
        # Discretize velocity: [-2.0, 2.0] -> [0, velocity_bins-1]
        velocity = state[1]
        vel_bin = int((velocity - self.env.v_range[0]) / 
                      (self.env.v_range[1] - self.env.v_range[0]) * self.velocity_bins)
        vel_bin = max(0, min(self.velocity_bins - 1, vel_bin))
        
        return (pos_bin, vel_bin)
    
    def get_action(self, state, training=True):
        """Select an action using epsilon-greedy policy."""
        discretized_state = self._discretize_state(state)
        
        # Explore: choose a random action
        if training and random.random() < self.exploration_rate:
            return random.randint(0, 2)  # Random action (0, 1, or 2)
        
        # Exploit: choose the best action from Q-table
        if discretized_state not in self.q_table:
            # If state not in Q-table, initialize with zeros
            self.q_table[discretized_state] = [0.0, 0.0, 0.0]
        
        return self.q_table[discretized_state].index(max(self.q_table[discretized_state]))
    
    def update(self, state, action, reward, next_state, done, training=True):
        """Update Q-values using the Q-learning update rule."""
        discretized_state = self._discretize_state(state)
        discretized_next_state = self._discretize_state(next_state)
        
        # Initialize Q-values if not exist
        if discretized_state not in self.q_table:
            self.q_table[discretized_state] = [0.0, 0.0, 0.0]
        
        if discretized_next_state not in self.q_table:
            self.q_table[discretized_next_state] = [0.0, 0.0, 0.0]
        
        # Q-learning update rule
        current_q = self.q_table[discretized_state][action]
        max_next_q = max(self.q_table[discretized_next_state])
        
        # Calculate new Q-value
        if done:
            # If done, there is no next state
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            # Standard Q-learning formula
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[discretized_state][action] = new_q
        
        # Decay exploration rate
        if done and training:
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay)
    
    def train(self, num_episodes=1000, max_steps=100, render_every=100):
        """Train the agent on the environment."""
        rewards_per_episode = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Select action
                action = self.get_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update Q-table
                self.update(state, action, reward, next_state, done, training=True)
                
                # Render if needed
                if render_every > 0 and episode % render_every == 0:
                    self.env.render()
                    time.sleep(0.1)  # Slow down rendering
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            rewards_per_episode.append(total_reward)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = sum(rewards_per_episode[-100:]) / min(100, len(rewards_per_episode))
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Exploration Rate: {self.exploration_rate:.2f}")
        
        return rewards_per_episode
    
    def evaluate(self, num_episodes=10, render=True):
        """Evaluate the trained agent."""
        total_rewards = 0
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action (no exploration during evaluation)
                action = self.get_action(state, training=False)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if render:
                    self.env.render()
                    time.sleep(0.2)  # Slower rendering for evaluation
                
                state = next_state
                episode_reward += reward
                
                if info.get("is_parked", False):
                    success_count += 1
            
            total_rewards += episode_reward
            
        avg_reward = total_rewards / num_episodes
        success_rate = success_count / num_episodes
        
        print(f"Evaluation over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2%}")
        
        return avg_reward, success_rate


def main():
    """Test the Q-learning agent."""
    env = OneDimParkingEnv(render_mode="human")
    agent = QLearningAgent(env)
    
    print("Training Q-learning agent...")
    agent.train(num_episodes=1000, render_every=0)
    
    print("\nEvaluating agent performance...")
    agent.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()