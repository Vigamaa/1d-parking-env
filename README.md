# OneDimParkingEnv 🚗

A simple reinforcement learning environment simulating a car that must stop inside a target parking zone.

## Overview

This is a custom Gymnasium environment that simulates a car moving in one dimension (along a straight line). The agent controls the car's acceleration and must learn to park the car (stop with zero velocity) inside a designated parking zone.

The environment is designed to be compatible with standard reinforcement learning algorithms and frameworks like stable-baselines3.

## Environment Details

### State Space
- Position (x): Range from -5.0 to 5.0
- Velocity (v): Range from -2.0 to 2.0

### Action Space
- 0: Accelerate backward (-1.0 m/s²)
- 1: Coast (0.0 m/s²)
- 2: Accelerate forward (+1.0 m/s²)

### Reward Function
- +1.0: Stopped in the center zone (i.e., parked successfully)
- +0.5: In the parking zone but still moving
- -0.1: Far away or moving past zone
- -10.0: Crashed (out of bounds)

### Episode Termination
- Car is parked (stopped in the target zone)
- Car crashes (goes outside the allowed area)
- Maximum steps reached (100 steps)

## Installation

```bash
# Install dependencies
pip install gymnasium numpy stable-baselines3
