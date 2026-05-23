# Deep Q-Network (DQN) for Lunar Lander

A PyTorch implementation of Deep Q-Network for the LunarLander-v3 environment using experience replay and target networks.

## Overview

This project trains a DQN agent to land a lunar lander on the moon's surface safely. The agent learns through reinforcement learning by trial and error.

**Goal:** Achieve average score of 200+ over 100 episodes

## Environment

**LunarLander-v3** from OpenAI Gymnasium

- **State Space:** 8 continuous values (position, velocity, angle, etc.)
- **Action Space:** 4 discrete actions (do nothing, left engine, main engine, right engine)
- **Reward:** +100 to +140 for safe landing, -100 for crash, penalties for fuel use

## Installation

### Step 1: Create Virtual Environment
```bash
python -m venv lunar_env
# Windows:
lunar_env\Scripts\activate
# Linux/Mac:
source lunar_env/bin/activate
```

### Step 2: Install PyTorch
```bash
# GPU (recommended):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# CPU only:
pip install torch torchvision torchaudio
```

### Step 3: Install Dependencies
```bash
pip install gymnasium gymnasium[box2d] imageio imageio-ffmpeg matplotlib numpy
```

### Step 4: Install Jupyter
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=lunar_env --display-name "Python (Lunar Lander)"
```

## Usage

### Training
```python
from core.vanilla_dqn_lunar_lander import train_dqn, evaluate_dqn

# Train agent
agent, rewards, losses = train_dqn(num_episodes=500)

# Evaluate
eval_rewards = evaluate_dqn(agent, num_episodes=10)
```

### Save Results
```python
from core.vanilla_dqn_lunar_lander import save_results, save_model, save_plot

# Define hyperparameters
hyperparameters = {
    'learning_rate': 0.001,
    'num_episodes': 500,
    'num_layers': 2,
    'hidden_dim': 128,
    'epsilon_decay': 0.995,
    'batch_size': 32,
    'gamma': 0.99
}

# Save all results
save_results(agent, rewards, losses, hyperparameters, experiment_name='baseline')
save_model(agent, experiment_name='baseline')
save_plot(rewards, losses, experiment_name='baseline')
```

## Experiment Tracking

### The Three Save Functions

#### **1. save_results()** - Save Numerical Data
```python
save_results(agent, rewards, losses, hyperparameters,
             experiment_name='baseline')
```

**What it saves:**
- Trained agent object (full DQN)
- Episode rewards list
- Episode losses
- Hyperparameters dictionary
- Summary metrics (best reward, average reward, etc.)

**File location:** `results/baseline/results.pkl`

**Use it:** After training, to save all numerical results for analysis or loading the trained agent later

**Example:**
```python
hyperparameters = {
    'learning_rate': 0.001,
    'num_episodes': 500,
    'batch_size': 32,
    'num_layers': 2,
    'hidden_dim': 128,
    'epsilon_decay': 0.995,
    'gamma': 0.99
}

save_results(agent, rewards, losses, hyperparameters, experiment_name='baseline')
```

---

#### **2. save_model()** - Save Model Weights Only
```python
save_model(agent, experiment_name='baseline')
```

**What it saves:**
- Only the trained neural network weights
- Much smaller file (~1-2 MB)

**File location:** `results/baseline/model.pt`

**Use it:** When you want a lightweight version to reload and evaluate later

**Load it later:**
```python
from core.vanilla_dqn_lunar_lander import load_model

agent = load_model(state_dim=8, action_dim=4, 
                   experiment_name='baseline',
                   num_layers=2, hidden_dim=128)
eval_rewards = evaluate_dqn(agent, num_episodes=10)
```

---

#### **3. save_plot()** - Save Training Visualization
```python
save_plot(rewards, losses, experiment_name='baseline')
```

**What it saves:**
- Training rewards curve (raw + smoothed line)
- Training loss curve
- PNG image file for easy sharing

**File location:** `results/baseline/plot.png`

**Use it:** Always! Creates a visual record of your training progress

### Complete Workflow: Train and Save

```python
from core.vanilla_dqn_lunar_lander import train_dqn, save_results, save_model, save_plot

# Step 1: Train your agent
print("Training agent...")
agent, rewards, losses = train_dqn(num_episodes=500, learning_rate=0.001)

# Step 2: Define hyperparameters
hyperparameters = {
    'learning_rate': 0.001,
    'num_episodes': 500,
    'batch_size': 32,
    'num_layers': 2,
    'hidden_dim': 128,
    'epsilon_decay': 0.995,
    'gamma': 0.99
}

# Step 3: Save everything
save_results(agent, rewards, losses, hyperparameters, experiment_name='baseline')
save_model(agent, experiment_name='baseline')
save_plot(rewards, losses, experiment_name='baseline')

print("Done! Check results/baseline/")
```

**Creates three files:**
```
results/baseline/
├── results.pkl      ← Full agent + metrics
├── model.pt         ← Just weights
└── plot.png         ← Training visualization
```

---

### Running Ablation Studies

Test different hyperparameter values:

```python
learning_rates = [0.0001, 0.001, 0.01]

for lr in learning_rates:
    agent, rewards, losses = train_dqn(num_episodes=100, learning_rate=lr)
    
    hyperparameters = {
        'learning_rate': lr,
        'num_layers': 2,
        'hidden_dim': 128,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'num_episodes': 100
    }
    
    # Results saved to: results/ablations/learning_rate/{lr}/
    save_results(agent, rewards, losses, hyperparameters,
                 experiment_name='learning_rate', variant_name=str(lr))
    save_model(agent, experiment_name='learning_rate', variant_name=str(lr))
    save_plot(rewards, losses, experiment_name='learning_rate', variant_name=str(lr))
```

### Folder Structure

```
results/
├── baseline/
│   ├── results.pkl
│   ├── model.pt
│   └── plot.png
└── ablations/
    ├── learning_rate/
    │   ├── 0.0001/
    │   │   ├── results.pkl
    │   │   ├── model.pt
    │   │   └── plot.png
    │   └── 0.001/
    └── num_layers/
        ├── 1/
        ├── 2/
        └── 3/
```

### Load and Compare Results

```python
import pickle
import numpy as np

# Load results
with open('results/ablations/learning_rate/0.001/results.pkl', 'rb') as f:
    results = pickle.load(f)

final_avg = np.mean(results['episode_rewards'][-25:])
best_reward = results['best_reward']
hyperparams = results['hyperparameters']

print(f"Final Average: {final_avg:.2f}, Best: {best_reward:.2f}")
```

## Project Structure

```
COMP3242-Group-Project-DQN/
├── README.md
├── core/
│   └── vanilla_dqn_lunar_lander.py    # Main DQN implementation
├── experiments/
│   ├── baseline_dqn.ipynb
│   ├── dqn_learning_rate_ablation.ipynb
│   ├── dqn_num_layers_ablation.ipynb
│   └── dqn_epsilon_decay_ablation.ipynb
└── results/                           # Created automatically when you save
    ├── baseline/
    └── ablations/
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 0.001 | Adam optimizer learning rate |
| num_episodes | 500 | Training episodes |
| batch_size | 32 | Samples per training step |
| num_layers | 2 | Number of hidden layers |
| hidden_dim | 128 | Hidden layer size |
| epsilon_decay | 0.995 | Exploration decay rate |
| gamma | 0.99 | Discount factor |
| buffer_capacity | 10000 | Experience replay buffer size |

## Troubleshooting

**GPU not being used:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:
```bash
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Missing module error:**
```bash
pip install gymnasium gymnasium[box2d] imageio imageio-ffmpeg matplotlib numpy
```

**Jupyter kernel issue:**
Change kernel in Jupyter: Kernel → Change Kernel → Python (Lunar Lander)

## References

- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Lunar Lander Environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

---

**Happy Landing! 🚀🌙**
