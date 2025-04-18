# 1D Parking Environment 🚗

A reinforcement learning (RL) environment simulating a car that must stop precisely inside a target parking zone, with both web-based and console interfaces.

## Overview

This project implements a custom Gymnasium-inspired environment for a one-dimensional car parking task. In this environment:

- The car moves along a linear track (1D) from -5 to +5
- The agent controls the car's acceleration (backward, coast, forward)
- The goal is to park precisely in a target zone in the middle of the track
- The agent receives rewards based on parking success or failure

The project includes:
- The core environment implementation
- A Q-learning agent implementation
- A web interface with manual play and RL simulation modes
- A console-based manual control mode

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

## Installation and Setup

### Method 1: Clone and Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/1d-parking-env.git
cd 1d-parking-env

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you don't have a requirements.txt file, you can:

1. Use the provided dependencies.txt file:
```bash
pip install -r dependencies.txt
```

2. Or install packages manually:
```bash
pip install flask gymnasium flask-sqlalchemy psycopg2-binary gunicorn
```

### Method 2: Run on Replit

1. Visit the Replit page: [https://replit.com/@YourUsername/1d-parking-env](https://replit.com/@YourUsername/1d-parking-env)
2. Fork the repl to your account
3. Press the Run button to start the application

### Creating requirements.txt

If you need to generate a requirements.txt file before uploading to GitHub, run:

```bash
pip freeze > requirements.txt
```

This will capture all installed packages. For a minimal requirements file, you can create one manually:

```
flask==2.3.2
gymnasium==0.28.1
flask-sqlalchemy==3.0.5
gunicorn==21.2.0
psycopg2-binary==2.9.6
```

### Environment Variables (Optional)

For production deployments, you might want to set:

```
FLASK_ENV=production
SECRET_KEY=your_secret_key
```

## Usage

There are multiple ways to interact with the environment:

### 1. Web Interface (Recommended)

Start the web server:

```bash
python main.py
# or use gunicorn for production:
gunicorn --bind 0.0.0.0:5000 main:app
```

Then open a browser and navigate to:
- Local: `http://localhost:5000`
- On Replit: The URL shown in the webview

The web interface provides:
- **Main Page**: Navigation between modes
- **Manual Play Mode**: Control the car yourself with keyboard inputs
- **RL Simulation Mode**: Watch and train a Q-learning agent that learns to park

### 2. Console-Based Manual Control

For a simple terminal-based interface, run:

```bash
python manual_play.py
```

This allows you to control the car using keyboard inputs:
- `A`: Accelerate backward
- `S`: Coast (no acceleration)
- `D`: Accelerate forward
- `Q`: Quit

### 3. Test Script

To verify the environment is working correctly:

```bash
python test_env.py
```

### 4. Direct RL Agent Training

To train a Q-learning agent directly from the command line:

```bash
python rl_agent.py
```

## RL Agent Features

The Q-learning agent implementation:
- Discretizes the continuous state space
- Uses an epsilon-greedy exploration policy
- Learns through trial and error to master the parking task
- Adapts its exploration rate as it learns
- Provides detailed metrics on performance during training

### Learning Metrics Explained

In the RL Simulation mode, you'll see various metrics that help you understand the agent's learning progress:

- **Success Rate**: Percentage of episodes where the agent successfully parked
- **Average Reward**: Mean reward obtained across all training episodes
- **Exploration Rate (ε)**: Controls the balance between exploration (trying new actions) and exploitation (using learned knowledge)
- **States Discovered**: Number of unique states the agent has encountered and learned values for
- **Learning Progress Bar**: Visual indicator of training progress (based on success rate)
- **Episode Results**: Visual history of the last 10 episodes (success/failure)

The agent is considered to have mastered the task when it achieves a success rate of 80% or higher. The training process will alert you when the agent reaches this milestone.

## Web-Based Environment Visualization

The web interface provides an enhanced visualization of the environment:

- **Track**: Graphical representation of the 1D track
- **Car**: Animated car with position updates
- **Parking Zone**: Clearly marked target area
- **Velocity Indicator**: Dynamic arrow showing direction and magnitude of velocity
- **Real-Time Stats**: Position, velocity, and reward updates
- **Learning Metrics**: Comprehensive display of RL agent performance (in simulation mode)

All visualizations are responsive and work on both desktop and mobile browsers.

## File Structure

- `one_dim_parking_env.py`: Core environment implementation
- `rl_agent.py`: Q-learning agent implementation
- `manual_play.py`: Console-based manual control
- `test_env.py`: Simple environment testing
- `main.py`: Web interface and API endpoints
- `templates/`: HTML templates for web interface
  - `main_page.html`: Landing page with mode selection
  - `manual_play.html`: Interactive manual control interface
  - `rl_simulation.html`: RL agent training and visualization interface

## Customization

You can modify the environment parameters in `one_dim_parking_env.py`:
- Track length and boundaries
- Target zone size and position
- Car dynamics (acceleration limits, friction)
- Reward structure

For the RL agent, adjust hyperparameters in `rl_agent.py`:
- Learning rate (alpha)
- Discount factor (gamma)
- Exploration rate (epsilon) and decay
- State discretization granularity

## Troubleshooting

### Common Issues

**Issue**: Web server won't start
**Solution**: Check for port conflicts. Try changing the port in the gunicorn command or use `python main.py` instead.

**Issue**: Keyboard controls not working in manual mode
**Solution**: Click on the game area first to ensure it has focus before using keyboard controls.

**Issue**: RL agent isn't learning effectively
**Solutions**:
- Increase the number of training episodes (try 100+)
- Adjust the learning rate (try values between 0.05 and 0.2)
- Increase state discretization (modify `positionBins` and `velocityBins` in the RL agent)

**Issue**: Missing dependencies
**Solution**: Ensure all required packages are installed using `pip install flask gymnasium flask-sqlalchemy gunicorn` or from requirements.txt.

### Debugging

If you encounter any issues, you can enable additional debug output:

1. In `main.py`, add:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Run the web server with debug mode:
   ```python
   # In main.py
   if __name__ == '__main__':
       app.run(debug=True, host='0.0.0.0', port=5000)
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
