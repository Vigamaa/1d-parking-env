#!/usr/bin/env python3
"""
Web interface for the 1D Parking Environment.
"""

from flask import Flask, render_template, request, jsonify
from one_dim_parking_env import OneDimParkingEnv
import json

app = Flask(__name__)
env = None

@app.route('/')
def index():
    """Main page handler."""
    return render_template('index.html')

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
    
    # Render the environment as an ASCII graphic
    env_render = env.render() if env.render_mode == "ansi" else None
    if env_render is None:
        # Create a simple text representation
        track = "-" * 50
        pos_idx = int((env.x - env.x_range[0]) / (env.x_range[1] - env.x_range[0]) * 49)
        target_start = int((env.target_zone[0] - env.x_range[0]) / (env.x_range[1] - env.x_range[0]) * 49)
        target_end = int((env.target_zone[1] - env.x_range[0]) / (env.x_range[1] - env.x_range[0]) * 49)
        
        visual = list(track)
        # Add target zone
        for i in range(target_start, target_end + 1):
            visual[i] = "P"
        # Add car position
        visual[pos_idx] = "C"
        
        env_render = "".join(visual)
    
    return jsonify({
        'observation': observation,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated,
        'info': info,
        'render': env_render
    })

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
        #environment-display {
            font-family: monospace;
            margin-top: 20px;
            font-size: 18px;
            white-space: pre;
        }
        #actions {
            margin-top: 20px;
        }
        #status {
            margin-top: 20px;
            padding: 10px;
            background-color: var(--bs-dark);
        }
        .game-container {
            max-width: 800px;
        }
    </style>
</head>
<body>
    <div class="container game-container">
        <h1>1D Parking Environment</h1>
        <p class="lead">
            Use the controls to park the car (C) in the parking zone (P).
            <br>
            Try to stop the car within the parking zone with close to zero velocity.
        </p>
        
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Controls</h5>
                <button id="reset" class="btn btn-warning">Reset Environment</button>
                <div id="actions" class="btn-group" role="group">
                    <button id="action-0" class="btn btn-primary">Accelerate Backward</button>
                    <button id="action-1" class="btn btn-secondary">Coast</button>
                    <button id="action-2" class="btn btn-primary">Accelerate Forward</button>
                </div>
            </div>
        </div>
        
        <div id="environment-display" class="card p-3 mt-3"></div>
        
        <div id="status" class="card mt-3">
            <h5>Status</h5>
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
            <div class="mt-2">
                <p><strong>Message:</strong> <span id="message"></span></p>
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
            totalSteps: 0
        };
        
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
                
                // Update display
                updateDisplay(data.render);
                
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
                    document.querySelectorAll('#actions button').forEach(btn => {
                        btn.disabled = true;
                    });
                }
            } catch (error) {
                console.error('Error taking step:', error);
                document.getElementById('message').textContent = 'Error taking step in environment.';
            }
        }
        
        // Update the environment display
        function updateDisplay(render) {
            document.getElementById('environment-display').textContent = render;
        }
        
        // Event listeners
        document.getElementById('reset').addEventListener('click', function() {
            resetEnvironment();
            // Re-enable action buttons
            document.querySelectorAll('#actions button').forEach(btn => {
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
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', resetEnvironment);
    </script>
</body>
</html>
""")
    
    # Create an instance of the environment with ansi render mode
    env = OneDimParkingEnv(render_mode="ansi")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000)