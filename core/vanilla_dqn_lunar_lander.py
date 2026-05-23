import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from gymnasium import make
import pickle
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Q-Network Architecture
class QNetwork(nn.Module):
    """Simple neural network for Q-value approximation with variable layers"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super(QNetwork, self).__init__()
        self.num_layers = num_layers

        # Build layers dynamically
        layers = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

#Experience replay buffer
class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

#DQN Agent
class VanillaDQN:
    """Vanilla DQN Agent"""
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, hidden_dim=128, buffer_capacity=10000,
                 num_layers=2):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_layers = num_layers

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self, batch_size):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Compute Q-values for chosen actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#Training function
def train_dqn(num_episodes=500, batch_size=32, update_frequency=4, target_update_frequency=100,
              num_layers=2, hidden_dim=128, epsilon_decay=0.995, learning_rate=1e-3):
    """Train the DQN agent on LunarLander"""

    # Create environment
    env = make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Create agent
    agent = VanillaDQN(state_dim, action_dim, num_layers=num_layers,
                       hidden_dim=hidden_dim, epsilon_decay=epsilon_decay, learning_rate=learning_rate)

    # Training loop
    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        step = 0

        while not done:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update network
            if step % update_frequency == 0:
                loss = agent.update(batch_size)
                if loss is not None:
                    episode_loss.append(loss)

            episode_reward += reward
            state = next_state
            step += 1

        # Update target network
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()

    return agent, episode_rewards, episode_losses

#Evaluate DQN
def evaluate_dqn(agent, num_episodes=10, render=False):
    """Evaluate the trained DQN agent"""
    env = make('LunarLander-v3', render_mode='human' if render else None)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage evaluation reward: {avg_reward:.2f}")

    return episode_rewards

#Plot Results
def plot_results(episode_rewards, episode_losses):
    """Plot training results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training: Episode Rewards')
    axes[0].grid(True, alpha=0.3)

    # Plot smoothed rewards
    window = min(50, len(episode_rewards) // 2)  # Use smaller window if not enough episodes
    if window > 1:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, window-1+len(smoothed_rewards)), smoothed_rewards, color='red',
                     label=f'Smoothed (window={window})', linewidth=2)
    axes[0].legend()

    # Plot loss
    if episode_losses:
        axes[1].plot(episode_losses, alpha=0.6)
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training: Loss')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Helper function for organized experiment paths
def get_experiment_path(experiment_name, variant_name=None, file_type='results'):
    """Generate organized experiment folder structure with absolute paths

    Works from any directory (experiments/, root, etc.)
    Searches upward to find the project root (where 'core' and 'experiments' folders exist)

    Example:
        get_experiment_path('baseline') -> '/path/to/CartPoleRL-Ablation/results/baseline/results.pkl'
    """
    # Method 1: Try using __file__ to locate the script
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # If we're in 'core/', go up one level
        if os.path.basename(script_dir) == 'core':
            project_root = os.path.dirname(script_dir)
        else:
            project_root = script_dir
    except:
        # Method 2: If __file__ doesn't work, search from current working directory
        project_root = os.getcwd()
        # Walk upward to find the project root (contains 'core' and 'experiments' folders)
        while not (os.path.exists(os.path.join(project_root, 'core')) and
                   os.path.exists(os.path.join(project_root, 'experiments'))):
            parent = os.path.dirname(project_root)
            if parent == project_root:  # Reached filesystem root
                break
            project_root = parent

    # Build results path
    if experiment_name == 'baseline':
        base_path = os.path.join(project_root, 'results', 'baseline')
    else:
        base_path = os.path.join(project_root, 'results', 'ablations', experiment_name, str(variant_name))

    file_ext = {
        'results': '.pkl',
        'model': '.pt',
        'plot': '.png'
    }.get(file_type, '')

    return os.path.join(base_path, f'{file_type}{file_ext}')


# Save and Load Results
def save_results(agent, episode_rewards, episode_losses, hyperparameters,
                 experiment_name='baseline', variant_name=None, save_path=None):
    """Save training results to pickle file

    Args:
        agent: trained DQN agent
        episode_rewards: list of episode rewards
        episode_losses: list of training losses
        hyperparameters: dict of hyperparameters used
        experiment_name: name of experiment (e.g., 'baseline', 'epsilon_decay', 'num_layers')
        variant_name: variant identifier (e.g., '0.995', '2', 'fast_decay')
        save_path: custom path (if not provided, auto-generated based on experiment)
    """
    if save_path is None:
        save_path = get_experiment_path(experiment_name, variant_name, 'results')

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_avg_reward': np.mean(episode_rewards[-25:]),
        'mean_reward': np.mean(episode_rewards),
        'best_reward': max(episode_rewards),
        'hyperparameters': hyperparameters
    }

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {save_path}")
    return results


def load_results(load_path='../results/baseline_results.pkl'):
    """Load training results from pickle file"""
    with open(load_path, 'rb') as f:
        results = pickle.load(f)

    print(f"Results loaded from {load_path}")
    return results


def save_model(agent, experiment_name='baseline', variant_name=None, save_path=None):
    """Save trained agent weights

    Args:
        agent: trained DQN agent
        experiment_name: name of experiment (e.g., 'baseline', 'epsilon_decay', 'num_layers')
        variant_name: variant identifier (e.g., '0.995', '2')
        save_path: custom path (if not provided, auto-generated)
    """
    if save_path is None:
        save_path = get_experiment_path(experiment_name, variant_name, 'model')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(agent.q_network.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(state_dim, action_dim, experiment_name='baseline', variant_name=None,
               num_layers=2, hidden_dim=128, load_path=None):
    """Load trained agent weights

    Args:
        state_dim: state dimension
        action_dim: action dimension
        experiment_name: name of experiment
        variant_name: variant identifier
        num_layers: number of layers (must match saved model)
        hidden_dim: hidden dimension (must match saved model)
        load_path: custom path (if not provided, auto-generated)
    """
    if load_path is None:
        load_path = get_experiment_path(experiment_name, variant_name, 'model')

    agent = VanillaDQN(state_dim, action_dim, num_layers=num_layers, hidden_dim=hidden_dim)
    agent.q_network.load_state_dict(torch.load(load_path, map_location=device))

    print(f"Model loaded from {load_path}")
    return agent


def save_plot(episode_rewards, episode_losses, experiment_name='baseline', variant_name=None,
              save_path=None):
    """Save training plot as image

    Args:
        episode_rewards: list of episode rewards
        episode_losses: list of training losses
        experiment_name: name of experiment (e.g., 'baseline', 'epsilon_decay', 'num_layers')
        variant_name: variant identifier (e.g., '0.995', '2')
        save_path: custom path (if not provided, auto-generated)
    """
    if save_path is None:
        save_path = get_experiment_path(experiment_name, variant_name, 'plot')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training: Episode Rewards')
    axes[0].grid(True, alpha=0.3)

    # Plot smoothed rewards
    window = min(50, len(episode_rewards) // 2)  # Use smaller window if not enough episodes
    if window > 1:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, window-1+len(smoothed_rewards)), smoothed_rewards, color='red',
                     label=f'Smoothed (window={window})', linewidth=2)
    axes[0].legend()

    # Plot loss
    if episode_losses:
        axes[1].plot(episode_losses, alpha=0.6)
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training: Loss')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

