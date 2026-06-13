# COMP3242 Group Project: DQN for LunarLander-v3

This repository contains our COMP3242 group project implementation and experiments for training a Deep Q-Network (DQN) agent on the `LunarLander-v3` environment.

The project studies how selected DQN hyperparameters affect training performance in a simple reinforcement learning environment. We use a one-at-a-time ablation design: epsilon decay, learning rate, and network depth are varied separately while the other hyperparameters are kept at a fixed baseline.

## Environment

We use `LunarLander-v3` from Gymnasium.

- State space: 8 continuous state values.
- Action space: 4 discrete actions.
- Objective: learn a landing policy that reaches high episode reward.
- Success reference: Gymnasium considers an average reward of 200 or above as solving the task.

## Repository Structure

```text
COMP3242-Group-Project-DQN/
├── README.md
├── requirements.txt
├── core/
│   └── vanilla_dqn_lunar_lander.py
├── experiments/
│   ├── baseline_dqn.ipynb
│   ├── epsilon_decay_ablation.ipynb
│   ├── lr_experiments.ipynb
│   ├── network_depth_ablation.ipynb
│   └── params_eval.ipynb
└── results/
    ├── baseline/
    └── ablations/
```

## Main Files

| File | Purpose |
| --- | --- |
| `core/vanilla_dqn_lunar_lander.py` | Core DQN implementation, including replay buffer, target network updates, training, evaluation, and result-saving helpers. |
| `experiments/baseline_dqn.ipynb` | Baseline DQN training run. |
| `experiments/epsilon_decay_ablation.ipynb` | Ablation study for epsilon decay. |
| `experiments/lr_experiments.ipynb` | Ablation study for learning rate. |
| `experiments/network_depth_ablation.ipynb` | Ablation study for number of hidden layers. |
| `experiments/params_eval.ipynb` | Result loading, summary statistics, and comparison plots. |

## Installation

Create and activate a virtual environment:

```bash
python -m venv lunar_env
source lunar_env/bin/activate
```

On Windows, activate it with:

```bash
lunar_env\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If Box2D fails to install, try upgrading pip first:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Running the Experiments

Start Jupyter:

```bash
jupyter notebook
```

Then run the notebooks in `experiments/`. A typical order is:

1. `baseline_dqn.ipynb`
2. `epsilon_decay_ablation.ipynb`
3. `lr_experiments.ipynb`
4. `network_depth_ablation.ipynb`
5. `params_eval.ipynb`

The notebooks save models, training curves, and numerical summaries under `results/`.

## Baseline Hyperparameters

| Hyperparameter | Baseline value |
| --- | ---: |
| Epsilon decay | 0.995 |
| Learning rate | 0.001 |
| Discount factor | 0.99 |
| Replay buffer | 10000 |
| Batch size | 32 |
| Number of episodes | 500 |
| Evaluation episodes | 10 |
| Target update frequency | 100 episodes |
| Number of hidden layers | 2 |
| Hidden dimension | 128 |

## Ablation Settings

The ablation study varies one hyperparameter at a time.

| Hyperparameter | Values tested |
| --- | --- |
| Epsilon decay | 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 0.995, 0.999 |
| Learning rate | 3e-3, 2e-3, 1e-3, 8e-4, 7e-4, 5e-4, 3e-4, 1e-4, 5e-5 |
| Number of hidden layers | 1, 2, 3, 4, 5 |

Each configuration is trained for 500 episodes. Multi-seed results are used to reduce dependence on a single lucky or unlucky run.

## Example Usage

```python
from core.vanilla_dqn_lunar_lander import train_dqn, evaluate_dqn

agent, rewards, losses = train_dqn(
    num_episodes=500,
    learning_rate=0.001,
    epsilon_decay=0.995,
    num_layers=2,
    hidden_dim=128,
)

eval_rewards = evaluate_dqn(agent, num_episodes=10)
print(sum(eval_rewards) / len(eval_rewards))
```

## Notes on Results

The `results/` directory contains saved experiment outputs used for analysis and report figures. Some files are large because trained model weights and per-configuration result objects are stored alongside plots.

When interpreting the results, we focus mainly on reward-based metrics. Loss curves are useful for checking optimization behaviour, but lower temporal-difference loss does not always imply a better landing policy.

## References

- Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015.
- Gymnasium LunarLander documentation: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- PyTorch documentation: https://pytorch.org/
