#!/usr/bin/env python3
"""
Web interface for the 1D Parking Environment.
"""

from flask import Flask, render_template, request, jsonify
from one_dim_parking_env import OneDimParkingEnv
from rl_agent import QLearningAgent
import json
import threading
import time

app = Flask(__name__)
env = None
agent = None
simulation_thread = None
simulation_running = False

@app.route('/')
def index():
    """Main page handler."""
    return render_template('main_page.html')

@app.route('/manual')
def manual_play():
    """Manual play mode."""
    return render_template('manual_play.html')

@app.route('/simulation')
def rl_simulation():
    """RL simulation mode."""
    return render_template('rl_simulation.html')

@app.route('/api/reset', methods=['POST'])
def reset_env():
    """Reset the environment."""
    global env
    if env is None:
        env = OneDimParkingEnv()
        
    seed = request.json.get('seed')
    if seed is not None:
        seed = int(seed)
        
    observation, info = env.reset(seed=seed)
    
    return jsonify({
        'observation': observation,
        'info': info
    })

@app.route('/api/step', methods=['POST'])
def step_env():
    """Take a step in the environment."""
    global env
    if env is None:
        return jsonify({'error': 'Environment not initialized. Call /api/reset first.'}), 400
        
    action = request.json.get('action')
    if action is None:
        return jsonify({'error': 'Action is required.'}), 400
        
    observation, reward, terminated, truncated, info = env.step(int(action))
    
    # Create a visual representation of the environment
    render_data = create_visual_representation(env)
    
    return jsonify({
        'observation': observation,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated,
        'info': info,
        'render': render_data['text_render'],
        'visual_data': render_data['visual_data']
    })

def create_visual_representation(env):
    """Create a visual representation of the environment."""
    # First try to get the built-in render if available
    env_render = env.render() if hasattr(env, 'render_mode') and env.render_mode == "ansi" else None
    
    # Create a simple text representation
    track_length = 50
    track = "-" * track_length
    pos_idx = int((env.x - env.x_range[0]) / (env.x_range[1] - env.x_range[0]) * (track_length - 1))
    target_start = int((env.target_zone[0] - env.x_range[0]) / (env.x_range[1] - env.x_range[0]) * (track_length - 1))
    target_end = int((env.target_zone[1] - env.x_range[0]) / (env.x_range[1] - env.x_range[0]) * (track_length - 1))
    
    visual = list(track)
    # Add target zone
    for i in range(target_start, target_end + 1):
        visual[i] = "P"
    # Add car position
    visual[pos_idx] = "C"
    
    text_render = "".join(visual)
    
    # Create data for a more sophisticated visual representation
    visual_data = {
        'track_length': track_length,
        'car_position': pos_idx / (track_length - 1),  # Normalized position (0-1)
        'car_velocity': env.v,
        'target_zone': [target_start / (track_length - 1), target_end / (track_length - 1)],  # Normalized (0-1)
        'normalized_position': (env.x - env.x_range[0]) / (env.x_range[1] - env.x_range[0]),
        'normalized_velocity': (env.v - env.v_range[0]) / (env.v_range[1] - env.v_range[0])
    }
    
    return {
        'text_render': text_render,
        'visual_data': visual_data
    }

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start training the RL agent."""
    global env, agent, simulation_thread, simulation_running
    
    # Stop any existing simulation
    if simulation_running:
        simulation_running = False
        if simulation_thread is not None:
            simulation_thread.join(timeout=1.0)
    
    # Initialize environment if needed
    if env is None:
        env = OneDimParkingEnv()
    
    # Initialize agent if needed
    if agent is None:
        agent = QLearningAgent(env)
    
    # Get training parameters
    episodes = request.json.get('episodes', 100)
    
    # Start training in a background thread
    simulation_running = True
    simulation_thread = threading.Thread(target=train_agent_thread, args=(episodes,))
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({'status': 'Training started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop any running simulation."""
    global simulation_running
    
    simulation_running = False
    return jsonify({'status': 'Simulation stopping'})

@app.route('/api/simulate', methods=['POST'])
def simulate_agent():
    """Simulate the trained agent."""
    global env, agent, simulation_thread, simulation_running
    
    # Stop any existing simulation
    if simulation_running:
        simulation_running = False
        if simulation_thread is not None:
            simulation_thread.join(timeout=1.0)
    
    # Initialize environment if needed
    if env is None:
        env = OneDimParkingEnv()
    
    # Make sure we have a trained agent
    if agent is None:
        agent = QLearningAgent(env)
        # Train with a small number of episodes to have something
        agent.train(num_episodes=50, render_every=0)
    
    # Start simulation in a background thread
    simulation_running = True
    simulation_thread = threading.Thread(target=simulate_agent_thread)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({'status': 'Simulation started'})

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get the current state of the environment."""
    global env, simulation_running
    
    if env is None:
        return jsonify({'error': 'Environment not initialized'}), 400
    
    # Create a visual representation
    render_data = create_visual_representation(env)
    
    return jsonify({
        'observation': [env.x, env.v],
        'simulation_running': simulation_running,
        'render': render_data['text_render'],
        'visual_data': render_data['visual_data']
    })

def train_agent_thread(episodes=100):
    """Train the RL agent in a background thread."""
    global env, agent, simulation_running
    
    try:
        print(f"Starting training for {episodes} episodes...")
        step_interval = 0.1  # seconds between steps for visualization
        
        total_rewards = []
        success_count = 0
        
        for episode in range(episodes):
            if not simulation_running:
                print("Training stopped early.")
                break
                
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done and simulation_running:
                # Get action from agent
                action = agent.get_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update agent
                agent.update(state, action, reward, next_state, done, training=True)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Check if successfully parked
                if done and info.get("is_parked", False):
                    success_count += 1
                
                # Pause briefly to allow visualization
                time.sleep(step_interval)
            
            total_rewards.append(episode_reward)
            
            # Print progress
            if episode % 10 == 0 or episode == episodes - 1:
                avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
                success_rate = success_count / (episode + 1)
                print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}")
        
        print("Training completed.")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        simulation_running = False

def simulate_agent_thread(num_episodes=10):
    """Simulate the trained agent in a background thread."""
    global env, agent, simulation_running
    
    try:
        print("Starting agent simulation...")
        step_interval = 0.2  # seconds between steps for visualization
        
        for episode in range(num_episodes):
            if not simulation_running:
                print("Simulation stopped early.")
                break
                
            state, _ = env.reset()
            done = False
            
            while not done and simulation_running:
                # Get action from agent (no exploration)
                action = agent.get_action(state, training=False)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update state
                state = next_state
                
                # Pause to allow visualization
                time.sleep(step_interval)
            
            # Brief pause between episodes
            time.sleep(0.5)
        
        print("Simulation completed.")
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        simulation_running = False

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create a simple HTML interface
    with open('templates/index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>1D Parking Environment</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
        }
        .environment-text {
            font-family: monospace;
            font-size: 18px;
            white-space: pre;
            margin-bottom: 15px;
        }
        .control-section {
            margin-top: 20px;
        }
        .status-section {
            margin-top: 20px;
            padding: 15px;
            background-color: var(--bs-dark);
            border-radius: 5px;
        }
        .game-container {
            max-width: 900px;
        }
        .canvas-container {
            position: relative;
            width: 100%;
            height: 120px;
            margin-bottom: 10px;
        }
        canvas {
            width: 100%;
            height: 100%;
            display: block;
            background-color: var(--bs-dark);
            border-radius: 5px;
        }
        .mode-toggle {
            margin-top: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--bs-gray-700);
        }
        .training-controls {
            margin-top: 20px;
            display: none;
        }
        .manual-controls {
            margin-top: 20px;
        }
        .badge {
            margin-left: 10px;
        }
        .velocity-indicator {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container game-container">
        <h1>1D Parking Environment</h1>
        <p class="lead">
            Park a car on a one-dimensional track. The goal is to stop the car precisely in the 
            parking zone with minimal velocity.
        </p>
        
        <div class="card">
            <div class="card-body">
                <div class="mode-toggle">
                    <h5 class="card-title">Mode Selection</h5>
                    <div class="btn-group" role="group">
                        <button id="manual-mode" class="btn btn-primary active">Manual Control</button>
                        <button id="rl-mode" class="btn btn-secondary">RL Simulation</button>
                    </div>
                    <span id="simulation-badge" class="badge bg-success d-none">Simulation Running</span>
                    <div class="alert alert-info mt-3">
                        <strong>Tip:</strong> Click "RL Simulation" to switch to the AI agent mode!
                    </div>
                </div>
                
                <div id="manual-controls" class="manual-controls">
                    <h5 class="card-title">Manual Controls</h5>
                    <button id="reset" class="btn btn-warning">Reset Environment</button>
                    <div class="btn-group mt-3" role="group">
                        <button id="action-0" class="btn btn-primary">⬅️ Accelerate Backward</button>
                        <button id="action-1" class="btn btn-secondary">Coast</button>
                        <button id="action-2" class="btn btn-primary">Accelerate Forward ➡️</button>
                    </div>
                </div>
                
                <div id="training-controls" class="training-controls">
                    <h5 class="card-title">RL Agent Controls</h5>
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <label for="episodes" class="form-label">Training Episodes:</label>
                            <input type="number" id="episodes" class="form-control" value="100" min="10" max="1000">
                        </div>
                        <div class="col-md-6 mt-3">
                            <button id="train-agent" class="btn btn-success">Train Agent</button>
                            <button id="simulate-agent" class="btn btn-primary">Run Simulation</button>
                            <button id="stop-simulation" class="btn btn-danger">Stop</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-body">
                <h5 class="card-title">Environment Visualization</h5>
                <div class="canvas-container">
                    <canvas id="env-canvas"></canvas>
                    <div class="velocity-indicator" id="velocity-indicator"></div>
                </div>
                <div id="environment-display" class="environment-text"></div>
            </div>
        </div>
        
        <div id="status" class="card mt-3">
            <div class="card-body">
                <h5 class="card-title">Status</h5>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Position:</strong> <span id="position">0.0</span></p>
                        <p><strong>Velocity:</strong> <span id="velocity">0.0</span></p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Last Reward:</strong> <span id="reward">0.0</span></p>
                        <p><strong>In Target Zone:</strong> <span id="in-zone">No</span></p>
                    </div>
                </div>
                <div class="alert alert-info mt-2" role="alert">
                    <p><strong>Message:</strong> <span id="message">Initialize the environment to begin.</span></p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Game state
        let gameState = {
            observation: [0, 0],
            terminated: false,
            truncated: false,
            info: {},
            reward: 0,
            totalSteps: 0,
            simulation: {
                running: false,
                interval: null
            }
        };
        
        // Canvas setup
        const canvas = document.getElementById('env-canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas dimensions for proper resolution
        function resizeCanvas() {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }
        
        // Initial resize and listen for window resize
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Reset the environment
        async function resetEnvironment() {
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                gameState.observation = data.observation;
                gameState.info = data.info;
                gameState.terminated = false;
                gameState.truncated = false;
                gameState.reward = 0;
                gameState.totalSteps = 0;
                
                document.getElementById('message').textContent = 'Environment reset. Good luck!';
                
                // Take a null step to get the initial render
                await takeStep(1); // Coast on first step
            } catch (error) {
                console.error('Error resetting environment:', error);
                document.getElementById('message').textContent = 'Error resetting environment.';
            }
        }
        
        // Take a step in the environment
        async function takeStep(action) {
            if (gameState.terminated || gameState.truncated) {
                document.getElementById('message').textContent = 'Game over. Reset to start a new game.';
                return;
            }
            
            try {
                const response = await fetch('/api/step', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action })
                });
                
                const data = await response.json();
                
                gameState.observation = data.observation;
                gameState.reward = data.reward;
                gameState.terminated = data.terminated;
                gameState.truncated = data.truncated;
                gameState.info = data.info;
                gameState.totalSteps++;
                
                // Update both displays
                updateTextDisplay(data.render);
                updateCanvasDisplay(data.visual_data);
                
                // Update status
                document.getElementById('position').textContent = data.observation[0].toFixed(2);
                document.getElementById('velocity').textContent = data.observation[1].toFixed(2);
                document.getElementById('reward').textContent = data.reward.toFixed(2);
                document.getElementById('in-zone').textContent = data.info.in_target_zone ? 'Yes' : 'No';
                
                // Game over check
                if (data.terminated || data.truncated) {
                    let message = '';
                    if (data.info.is_parked) {
                        message = 'Success! Car parked correctly.';
                    } else if (data.observation[0] <= -5 || data.observation[0] >= 5) {
                        message = 'Crashed! Car went out of bounds.';
                    } else {
                        message = 'Time limit reached without parking.';
                    }
                    document.getElementById('message').textContent = message;
                    
                    // Disable action buttons
                    document.querySelectorAll('#manual-controls button').forEach(btn => {
                        if (btn.id !== 'reset') {
                            btn.disabled = true;
                        }
                    });
                }
            } catch (error) {
                console.error('Error taking step:', error);
                document.getElementById('message').textContent = 'Error taking step in environment.';
            }
        }
        
        // Simple client-side implementation of Q-learning
        const clientRL = {
            // Parameters
            learningRate: 0.1,         // Alpha
            discountFactor: 0.99,      // Gamma
            explorationRate: 0.3,      // Epsilon
            positionBins: 10,          // Number of discrete position bins
            velocityBins: 10,          // Number of discrete velocity bins
            qTable: {},                // Q-table (state -> [action values])
            
            // Discretize state for Q-table lookup
            discretizeState: function(state) {
                // Map position from [-5,5] to [0,positionBins-1]
                const position = state[0];
                const positionRange = 10; // -5 to 5 = 10 units
                let posBin = Math.floor((position + 5) / positionRange * this.positionBins);
                posBin = Math.max(0, Math.min(this.positionBins - 1, posBin));
                
                // Map velocity from [-2,2] to [0,velocityBins-1]
                const velocity = state[1];
                const velocityRange = 4; // -2 to 2 = 4 units
                let velBin = Math.floor((velocity + 2) / velocityRange * this.velocityBins);
                velBin = Math.max(0, Math.min(this.velocityBins - 1, velBin));
                
                return `${posBin},${velBin}`;
            },
            
            // Get action using epsilon-greedy policy
            getAction: function(state, training = true) {
                const discreteState = this.discretizeState(state);
                
                // Explore: choose random action
                if (training && Math.random() < this.explorationRate) {
                    return Math.floor(Math.random() * 3); // Random action (0, 1, or 2)
                }
                
                // Initialize state if not in Q-table
                if (!this.qTable[discreteState]) {
                    this.qTable[discreteState] = [0, 0, 0];
                }
                
                // Exploit: choose best action
                const actionValues = this.qTable[discreteState];
                let maxValue = actionValues[0];
                let bestAction = 0;
                
                for (let i = 1; i < actionValues.length; i++) {
                    if (actionValues[i] > maxValue) {
                        maxValue = actionValues[i];
                        bestAction = i;
                    }
                }
                
                return bestAction;
            },
            
            // Update Q-values
            update: function(state, action, reward, nextState, done) {
                const discreteState = this.discretizeState(state);
                const discreteNextState = this.discretizeState(nextState);
                
                // Initialize states if not in Q-table
                if (!this.qTable[discreteState]) {
                    this.qTable[discreteState] = [0, 0, 0];
                }
                
                if (!this.qTable[discreteNextState]) {
                    this.qTable[discreteNextState] = [0, 0, 0];
                }
                
                // Get current Q-value
                const currentQ = this.qTable[discreteState][action];
                
                // Get max Q-value for next state
                const nextValues = this.qTable[discreteNextState];
                const maxNextQ = Math.max(...nextValues);
                
                // Q-learning update rule
                let newQ;
                if (done) {
                    newQ = currentQ + this.learningRate * (reward - currentQ);
                } else {
                    newQ = currentQ + this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
                }
                
                // Update Q-table
                this.qTable[discreteState][action] = newQ;
            }
        };
        
        // Start training the agent (client-side implementation)
        async function startTraining() {
            const episodes = parseInt(document.getElementById('episodes').value) || 100;
            
            // Reset RL parameters for fresh training
            clientRL.explorationRate = 0.3;
            clientRL.qTable = {};
            
            // Show simulation badge
            gameState.simulation.running = true;
            document.getElementById('simulation-badge').classList.remove('d-none');
            document.getElementById('message').textContent = 'Training started. This may take some time...';
            
            // Run episodes in sequence
            let totalReward = 0;
            let successCount = 0;
            
            for (let episode = 0; episode < episodes; episode++) {
                // Skip if stopped
                if (!gameState.simulation.running) {
                    document.getElementById('message').textContent = 'Training stopped.';
                    break;
                }
                
                // Reset environment
                await resetEnvironment();
                let state = gameState.observation;
                let episodeReward = 0;
                let stepCount = 0;
                let done = false;
                
                // Run single episode
                while (!done && stepCount < 100 && gameState.simulation.running) {
                    // Get action from RL agent
                    const action = clientRL.getAction(state, true);
                    
                    // Take action
                    await takeStep(action);
                    
                    // Get new state and reward
                    const nextState = gameState.observation;
                    const reward = gameState.reward;
                    done = gameState.terminated || gameState.truncated;
                    
                    // Update Q-table
                    clientRL.update(state, action, reward, nextState, done);
                    
                    // Update state for next iteration
                    state = nextState;
                    episodeReward += reward;
                    stepCount++;
                    
                    // Check if parked successfully
                    if (done && gameState.info.is_parked) {
                        successCount++;
                    }
                    
                    // Small delay for visualization
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
                
                totalReward += episodeReward;
                
                // Update progress message every 10 episodes
                if (episode % 10 === 0 || episode === episodes - 1) {
                    const avgReward = totalReward / (episode + 1);
                    const successRate = successCount / (episode + 1) * 100;
                    document.getElementById('message').textContent = 
                        `Training progress: ${episode+1}/${episodes} episodes, ` +
                        `Avg Reward: ${avgReward.toFixed(2)}, Success Rate: ${successRate.toFixed(1)}%`;
                }
                
                // Decrease exploration rate over time
                clientRL.explorationRate = Math.max(0.05, clientRL.explorationRate * 0.95);
            }
            
            // Training complete
            if (gameState.simulation.running) {
                document.getElementById('message').textContent = 'Training completed! You can now run the simulation.';
                // Reduce exploration for evaluation
                clientRL.explorationRate = 0.05;
            }
            
            gameState.simulation.running = false;
            document.getElementById('simulation-badge').classList.add('d-none');
        }
        
        // Simulate the trained agent (client-side implementation)
        async function simulateAgent() {
            // Reset any ongoing simulation
            if (gameState.simulation.running) {
                stopSimulation();
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            gameState.simulation.running = true;
            document.getElementById('simulation-badge').classList.remove('d-none');
            document.getElementById('message').textContent = 'RL agent is now driving the car...';
            
            // Run 5 episodes of simulation
            const numEpisodes = 5;
            let totalReward = 0;
            let successCount = 0;
            
            for (let episode = 0; episode < numEpisodes; episode++) {
                if (!gameState.simulation.running) break;
                
                // Reset environment
                await resetEnvironment();
                let state = gameState.observation;
                let episodeReward = 0;
                let done = false;
                
                // Run single episode
                while (!done && gameState.simulation.running) {
                    // Get action from RL agent (no exploration during simulation)
                    const action = clientRL.getAction(state, false);
                    
                    // Take action
                    await takeStep(action);
                    
                    // Get new state and reward
                    const nextState = gameState.observation;
                    const reward = gameState.reward;
                    done = gameState.terminated || gameState.truncated;
                    
                    // Update state for next iteration
                    state = nextState;
                    episodeReward += reward;
                    
                    // Check if parked successfully
                    if (done && gameState.info.is_parked) {
                        successCount++;
                    }
                    
                    // Delay for better visualization
                    await new Promise(resolve => setTimeout(resolve, 200));
                }
                
                totalReward += episodeReward;
                
                // Brief pause between episodes
                if (episode < numEpisodes - 1 && gameState.simulation.running) {
                    document.getElementById('message').textContent = 
                        `Completed episode ${episode+1}/${numEpisodes}. Starting next episode...`;
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            // Simulation complete
            if (gameState.simulation.running) {
                const avgReward = totalReward / numEpisodes;
                const successRate = successCount / numEpisodes * 100;
                document.getElementById('message').textContent = 
                    `Simulation completed! Avg Reward: ${avgReward.toFixed(2)}, ` +
                    `Success Rate: ${successRate.toFixed(0)}%`;
            }
            
            gameState.simulation.running = false;
            document.getElementById('simulation-badge').classList.add('d-none');
        }
        
        // Stop the simulation
        function stopSimulation() {
            gameState.simulation.running = false;
            document.getElementById('simulation-badge').classList.add('d-none');
            document.getElementById('message').textContent = 'Simulation stopped.';
        }
        
        // Update the text display
        function updateTextDisplay(render) {
            document.getElementById('environment-display').textContent = render;
        }
        
        // Update the canvas display
        function updateCanvasDisplay(visualData) {
            if (!visualData) return;
            
            // Clear the canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const padding = 20;
            const trackHeight = 30;
            const trackY = canvas.height / 2 - trackHeight / 2;
            const trackWidth = canvas.width - padding * 2;
            
            // Draw the track
            ctx.fillStyle = '#444';
            ctx.fillRect(padding, trackY, trackWidth, trackHeight);
            
            // Draw the parking zone
            const parkingZoneStart = padding + visualData.target_zone[0] * trackWidth;
            const parkingZoneWidth = (visualData.target_zone[1] - visualData.target_zone[0]) * trackWidth;
            ctx.fillStyle = '#ffc107';  // Bootstrap warning color
            ctx.fillRect(parkingZoneStart, trackY, parkingZoneWidth, trackHeight);
            
            // Draw the car
            const carWidth = 30;
            const carHeight = 20;
            const carX = padding + visualData.car_position * trackWidth - carWidth / 2;
            const carY = trackY + trackHeight / 2 - carHeight / 2;
            
            // Draw car body
            ctx.fillStyle = '#0d6efd';  // Bootstrap primary color
            ctx.beginPath();
            ctx.roundRect(carX, carY, carWidth, carHeight, 5);
            ctx.fill();
            
            // Draw car windows
            ctx.fillStyle = '#aaddff';
            ctx.fillRect(carX + 5, carY + 4, 7, 5);
            ctx.fillRect(carX + 18, carY + 4, 7, 5);
            
            // Draw velocity indicator (arrow)
            const velocityIndicator = document.getElementById('velocity-indicator');
            const velocityCtx = document.createElement('canvas').getContext('2d');
            velocityCtx.canvas.width = canvas.width;
            velocityCtx.canvas.height = canvas.height;
            
            const velocity = visualData.car_velocity;
            const arrowLength = Math.min(Math.abs(velocity) * 50, 100);  // Scale velocity for display
            
            if (Math.abs(velocity) > 0.05) {  // Only show arrow if velocity is significant
                const arrowX = carX + carWidth / 2;
                const arrowY = carY - 10;
                const direction = velocity > 0 ? 1 : -1;
                
                // Draw the arrow
                velocityCtx.strokeStyle = velocity > 0 ? '#28a745' : '#dc3545';  // Green for forward, red for backward
                velocityCtx.lineWidth = 3;
                velocityCtx.beginPath();
                velocityCtx.moveTo(arrowX, arrowY);
                velocityCtx.lineTo(arrowX + direction * arrowLength, arrowY);
                
                // Arrow head
                velocityCtx.lineTo(arrowX + direction * arrowLength - direction * 10, arrowY - 5);
                velocityCtx.moveTo(arrowX + direction * arrowLength, arrowY);
                velocityCtx.lineTo(arrowX + direction * arrowLength - direction * 10, arrowY + 5);
                
                velocityCtx.stroke();
                
                // Update the velocity indicator
                velocityIndicator.innerHTML = '';
                velocityIndicator.appendChild(velocityCtx.canvas);
            } else {
                velocityIndicator.innerHTML = '';  // Clear when velocity is near zero
            }
        }
        
        // Mode switching event listeners
        document.getElementById('manual-mode').addEventListener('click', function() {
            document.getElementById('manual-controls').style.display = 'block';
            document.getElementById('training-controls').style.display = 'none';
            document.getElementById('manual-mode').classList.add('active', 'btn-primary');
            document.getElementById('manual-mode').classList.remove('btn-secondary');
            document.getElementById('rl-mode').classList.remove('active', 'btn-primary');
            document.getElementById('rl-mode').classList.add('btn-secondary');
            
            // Stop any running simulation
            if (gameState.simulation.running) {
                stopSimulation();
            }
            
            // Reset environment for manual control
            resetEnvironment();
        });
        
        document.getElementById('rl-mode').addEventListener('click', function() {
            document.getElementById('manual-controls').style.display = 'none';
            document.getElementById('training-controls').style.display = 'block';
            document.getElementById('rl-mode').classList.add('active', 'btn-primary');
            document.getElementById('rl-mode').classList.remove('btn-secondary');
            document.getElementById('manual-mode').classList.remove('active', 'btn-primary');
            document.getElementById('manual-mode').classList.add('btn-secondary');
            
            // Reset environment for RL mode
            resetEnvironment();
        });
        
        // Manual control event listeners
        document.getElementById('reset').addEventListener('click', function() {
            resetEnvironment();
            // Re-enable action buttons
            document.querySelectorAll('#manual-controls button').forEach(btn => {
                btn.disabled = false;
            });
        });
        
        document.getElementById('action-0').addEventListener('click', function() {
            takeStep(0); // Accelerate backward
        });
        
        document.getElementById('action-1').addEventListener('click', function() {
            takeStep(1); // Coast
        });
        
        document.getElementById('action-2').addEventListener('click', function() {
            takeStep(2); // Accelerate forward
        });
        
        // RL agent control event listeners
        document.getElementById('train-agent').addEventListener('click', startTraining);
        document.getElementById('simulate-agent').addEventListener('click', simulateAgent);
        document.getElementById('stop-simulation').addEventListener('click', stopSimulation);
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', resetEnvironment);
        
        // Add keyboard controls for manual mode
        document.addEventListener('keydown', function(event) {
            // Only process keys if in manual mode
            if (document.getElementById('manual-controls').style.display !== 'none') {
                if (event.key === 'a' || event.key === 'A' || event.key === 'ArrowLeft') {
                    takeStep(0); // Accelerate backward
                } else if (event.key === 's' || event.key === 'S') {
                    takeStep(1); // Coast
                } else if (event.key === 'd' || event.key === 'D' || event.key === 'ArrowRight') {
                    takeStep(2); // Accelerate forward
                } else if (event.key === 'r' || event.key === 'R') {
                    resetEnvironment();
                }
            }
        });
    </script>
</body>
</html>
""")
    
    # Create an instance of the environment with ansi render mode
    env = OneDimParkingEnv(render_mode="ansi")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000)