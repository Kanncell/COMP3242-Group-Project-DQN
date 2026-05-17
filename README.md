# Deep Q-Network (DQN) for Lunar Lander

A PyTorch implementation of Deep Q-Network (DQN) for solving the OpenAI Gymnasium LunarLander-v3 environment. This implementation uses **Standard DQN** with experience replay, target networks, and various optimization techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Environment](#environment)
- [Algorithm](#algorithm)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [GPU Support](#gpu-support)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Overview

This project implements a Deep Q-Network (DQN) agent to learn how to safely land a lunar lander on the moon's surface. The agent learns through trial and error using reinforcement learning, receiving rewards for successful landings and penalties for crashes.

### Goal
Train an agent to achieve an average score of **200+** over 100 consecutive episodes, indicating successful and efficient landings.

---

## 🌙 Environment

**Environment**: `LunarLander-v3` from OpenAI Gymnasium

### State Space
- **Dimension**: 8 continuous values
- **Features**:
  - Horizontal position (x)
  - Vertical position (y)
  - Horizontal velocity (x-dot)
  - Vertical velocity (y-dot)
  - Angle
  - Angular velocity
  - Left leg ground contact (boolean)
  - Right leg ground contact (boolean)

### Action Space
- **Dimension**: 4 discrete actions
- **Actions**:
  - `0`: Do nothing
  - `1`: Fire left orientation engine
  - `2`: Fire main engine (move up)
  - `3`: Fire right orientation engine

### Reward Structure
- **Landing safely**: +100 to +140 points
- **Crash**: -100 points
- **Fuel consumption**: Small penalty per engine fire
- **Distance from landing pad**: Small penalty
- **Episode ends**: When lander crashes, lands, or goes out of bounds

---

## 🧠 Algorithm

This implementation uses **Standard DQN** with the following components:

### 1. Q-Learning Update Rule

The agent learns by updating Q-values using the Bellman equation:

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

Where:
- `s`: current state
- `a`: action taken
- `r`: reward received
- `s'`: next state
- `a'`: next action
- `α`: learning rate
- `γ`: discount factor

### 2. Standard DQN Approach

**Target Q-value Calculation**:
```python
# Target network selects AND evaluates the best action
next_q_values = target_network(next_states).max(1)[0]
target = reward + gamma * next_q_values * (1 - done)
```

The target network is used for both:
1. **Action Selection**: Finding the best action (max Q-value)
2. **Action Evaluation**: Evaluating that action's Q-value

This is simpler than Double DQN but may lead to slight Q-value overestimation.

### 3. Key Components

#### Experience Replay
- Stores experiences `(state, action, reward, next_state, done)` in a replay buffer
- Breaks correlation between consecutive experiences
- Enables more efficient learning from past experiences
- **Buffer Size**: 100,000 experiences

#### Target Network
- Separate network with frozen weights
- Updated periodically (soft updates every step + hard updates every 100 episodes)
- Provides stable Q-value targets
- Prevents divergence during training

#### Epsilon-Greedy Exploration
- Balances exploration (random actions) vs exploitation (learned policy)
- **Initial ε**: 1.0 (100% exploration)
- **Final ε**: 0.01 (1% exploration)
- **Decay rate**: 0.996 per episode

---

## 🏗️ Architecture

### Neural Network Structure

```
Input Layer (8 neurons)
        ↓
Dense Layer (256 neurons) + LayerNorm + ReLU + Dropout(0.1)
        ↓
Dense Layer (256 neurons) + LayerNorm + ReLU + Dropout(0.1)
        ↓
Dense Layer (128 neurons) + LayerNorm + ReLU
        ↓
Output Layer (4 neurons - Q-values for each action)
```

### Network Features
- **Layer Normalization**: Stabilizes training by normalizing activations
- **Dropout (10%)**: Prevents overfitting
- **He Initialization**: Proper weight initialization for ReLU activations
- **Total Parameters**: ~165,000 trainable parameters

### Loss Function
- **Huber Loss (Smooth L1)**: More robust to outliers than MSE
- Combines benefits of L1 and L2 loss
- Less sensitive to extreme Q-value errors

---

## ✨ Features

### Training Enhancements
- ✅ **Experience Replay Buffer**: Efficient memory management with circular buffer
- ✅ **Target Network**: Stable Q-learning with soft + hard updates
- ✅ **Gradient Clipping**: Prevents exploding gradients (clip at 1.0)
- ✅ **Learning Rate Scheduler**: Reduces LR every 200 episodes (γ=0.9)
- ✅ **Early Stopping**: Stops when target score achieved
- ✅ **Best Model Checkpointing**: Saves model with highest average score

### Network Improvements
- ✅ **Deeper Architecture**: 4-layer network (256→256→128→4)
- ✅ **Layer Normalization**: Faster and more stable training
- ✅ **Dropout Regularization**: Reduces overfitting
- ✅ **AdamW Optimizer**: Adam with weight decay (1e-5)

### Monitoring & Visualization
- ✅ **Real-time Progress**: Episode scores, average scores, epsilon values
- ✅ **Training Plots**: Reward curves and loss curves with moving averages
- ✅ **Video Generation**: Records best landing attempts as MP4
- ✅ **Random Baseline**: Compares trained agent vs random actions

---

## 📦 Requirements

### Python Packages
```
torch>=2.5.1
torchvision>=0.20.1
torchaudio>=2.5.1
gymnasium>=0.29.0
gymnasium[box2d]
imageio>=2.31.0
imageio-ffmpeg>=0.4.9
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## 🚀 Installation

### Step 1: Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv lunar_lander_env

# Activate environment
# Windows:
lunar_lander_env\Scripts\activate
# Linux/Mac:
source lunar_lander_env/bin/activate
```

### Step 2: Install PyTorch with CUDA Support

For **GPU training** (recommended):

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

For **CPU-only** training:

```bash
pip install torch torchvision torchaudio
```

### Step 3: Install Dependencies

```bash
pip install gymnasium gymnasium[box2d] imageio imageio-ffmpeg matplotlib numpy
```

### Step 4: Verify GPU Support (Optional)

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output (if GPU is available):
```
CUDA Available: True
GPU: NVIDIA GeForce RTX XXXX
```

### Step 5: Install Jupyter (for notebook usage)

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=lunar_lander_env --display-name "Python (Lunar Lander)"
```

---

## 💻 Usage

### Training the Agent

#### Option 1: Run Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open: DQN_LunarLander_Improved.ipynb
# Select Kernel: "Python (Lunar Lander)"
# Run all cells
```

#### Option 2: Run as Python Script

```python
# In Python or IPython
from train_agent import train_agent, generate_video

# Train agent
agent, rewards, losses = train_agent(
    num_episodes=2000,
    target_score=200.0
)

# Generate video of best landing
generate_video(agent, num_attempts=10)
```

### Loading Pre-trained Model

```python
import torch
from agent import Agent

# Create agent
agent = Agent(state_size=8, action_size=4)

# Load trained weights
agent.local_qnetwork.load_state_dict(torch.load('checkpoint_best.pth'))
agent.local_qnetwork.eval()

# Test the agent
# (see generate_video function for evaluation code)
```

---

## ⚙️ Hyperparameters

### Network Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| Hidden Layer 1 | 256 | First hidden layer size |
| Hidden Layer 2 | 256 | Second hidden layer size |
| Hidden Layer 3 | 128 | Third hidden layer size |
| Dropout Rate | 0.1 | Dropout probability |

### Training Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 5e-4 | AdamW learning rate |
| Batch Size | 128 | Samples per training step |
| Gamma (γ) | 0.99 | Discount factor |
| Tau (τ) | 1e-3 | Soft update parameter |
| Weight Decay | 1e-5 | L2 regularization |

### Exploration Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Final exploration rate |
| Epsilon Decay | 0.996 | Decay rate per episode |

### Memory & Updates
| Parameter | Value | Description |
|-----------|-------|-------------|
| Replay Buffer Size | 100,000 | Maximum experiences stored |
| Update Frequency | 4 steps | Learn every N steps |
| Target Update (Soft) | Every step | Soft update frequency |
| Target Update (Hard) | 100 episodes | Hard update frequency |
| LR Scheduler Step | 200 episodes | Reduce LR every N episodes |

### Training
| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Episodes | 2000 | Maximum training episodes |
| Max Steps/Episode | 1000 | Maximum steps per episode |
| Target Score | 200.0 | Solving criteria (avg 100 eps) |

---

## 📊 Results

### Expected Performance

| Metric | Value |
|--------|-------|
| **Episodes to Solve** | 800-1200 episodes |
| **Final Average Score** | 200-250 points |
| **Training Time (GPU)** | 30-50 minutes |
| **Training Time (CPU)** | 2-4 hours |
| **Random Agent Score** | -200 to -400 points |

### Training Curves

The agent should show:
1. **Initial Phase (0-200 episodes)**: Negative scores, random exploration
2. **Learning Phase (200-800 episodes)**: Gradual improvement, scores increase
3. **Convergence (800+ episodes)**: Stable performance around 200-250 points

### Output Files

After training, the following files are generated:

- `checkpoint_best.pth` - Best performing model weights
- `checkpoint_final.pth` - Final model weights
- `training_rewards_graph.png` - Episode rewards over time
- `training_loss_graph.png` - Training loss over time
- `lunarlander_RandomActionsSim.mp4` - Random agent baseline video
- `lunarlander_TrainedSim.mp4` - Trained agent landing video



## 📁 Project Structure

```
lunar-lander-dqn/
│
├── DQN_LunarLander_Improved.ipynb   # Main training notebook
├── README.md                         # This file
│
├── Models/
│   ├── checkpoint_best.pth          # Best model weights
│   └── checkpoint_final.pth         # Final model weights
│
├── Visualizations/
│   ├── training_rewards_graph.png   # Rewards plot
│   ├── training_loss_graph.png      # Loss plot
│   ├── lunarlander_RandomActionsSim.mp4  # Random baseline
│   └── lunarlander_TrainedSim.mp4   # Trained agent
│
└── requirements.txt                  # Package dependencies
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'gymnasium'

```bash
pip install gymnasium gymnasium[box2d]
```

#### 2. ModuleNotFoundError: No module named 'imageio'

```bash
pip install imageio imageio-ffmpeg
```

#### 3. ValueError: Could not find a backend to open MP4

```bash
pip install imageio-ffmpeg
```

#### 4. CUDA out of memory

Reduce batch size:
```python
self.batch_size = 64  # instead of 128
```

#### 5. Training not using GPU (shows "Using device: cpu")

Check PyTorch installation:
```bash
python -c "import torch; print(torch.__version__)"
```

If it shows `+cpu`, reinstall with CUDA support:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 6. Jupyter kernel using wrong Python environment

Register the correct kernel:
```bash
python -m ipykernel install --user --name=lunar_lander_env --display-name "Python (GPU)"
```

Then in Jupyter: Kernel → Change Kernel → Python (GPU)

#### 7. Training is very slow

- **Use GPU**: Follow GPU installation steps
- **Reduce episodes**: Train for fewer episodes (1000 instead of 2000)
- **Increase batch size**: Use 256 instead of 128 (if you have enough memory)

---

## 📚 References

### Papers
1. **Playing Atari with Deep Reinforcement Learning**  
   Mnih et al., 2013  
   https://arxiv.org/abs/1312.5602

### Resources
- OpenAI Gymnasium Documentation: https://gymnasium.farama.org/
- PyTorch Documentation: https://pytorch.org/docs/
- Lunar Lander Environment: https://gymnasium.farama.org/environments/box2d/lunar_lander/

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- OpenAI Gymnasium for the LunarLander environment
- DeepMind for the original DQN algorithm
- PyTorch team for the deep learning framework

---
**Happy Landing! 🚀🌙**