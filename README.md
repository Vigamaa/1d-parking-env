# OneDimParkingEnv 🚗

A simple reinforcement learning environment simulating a car that must stop inside a target parking zone.

## Overview

This project implements a custom Gymnasium environment for a one-dimensional car parking task. In this environment:

- The car moves along a linear track (1D) from -5 to +5
- The agent controls the car's acceleration (backward, coast, forward)
- The goal is to park precisely in a target zone in the middle of the track
- The agent receives rewards based on parking success or failure

This environment follows the Gymnasium API and is compatible with reinforcement learning frameworks like stable-baselines3.

## State & Action Space

- **State Space**: 2D continuous space
  - Position (x): [-5.0, 5.0]
  - Velocity (v): [-2.0, 2.0]

- **Action Space**: Discrete (3 options)
  - 0: Accelerate backward (-1.0 m/s²)
  - 1: Coast (0.0 m/s²)
  - 2: Accelerate forward (+1.0 m/s²)

## Reward System

- **+1.0**: Successfully parked (stopped in target zone)
- **+0.5**: In target zone but still moving
- **-0.1**: Not in target zone
- **-10.0**: Crashed (outside track bounds)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/onedim-parking-env.git
cd onedim-parking-env

# Install dependencies
pip install gymnasium numpy stable-baselines3
