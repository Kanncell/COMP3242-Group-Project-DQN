# Learning Rate Experiment Summary

This note is a temporary record for Ling's learning-rate experiments. The final report will be written in LaTeX; this file is only for quickly checking the experiment settings and results.

## Setup

All runs used the same DQN training setup based on `DQN_LunarLander_Improved`. The only changed variable was the learning rate.

- Environment: `LunarLander-v3`
- Maximum episodes per run: `2000`
- Maximum steps per episode: `1000`
- Early stopping: training stops when the recent 100-episode average score reaches `200`
- Baseline learning rate: `5e-4`

## Results

| Learning Rate | Solved? | Solved Episode | Final Avg Score | Best Avg Score | Training Time |
|---:|:---:|---:|---:|---:|---:|
| `1e-4` | No | N/A | 154.37 | 179.99 | 1249.4s |
| `3e-4` | Yes | 1307 | 200.42 | 200.42 | 612.3s |
| `5e-4` | Yes | 811 | 200.30 | 200.30 | 388.8s |
| `1e-3` | Yes | 587 | 201.50 | 201.50 | 259.6s |

## Short Conclusion

Among the tested learning rates, `1e-3` performed best. It solved the environment fastest, reaching the target average score in 587 episodes. The original baseline `5e-4` also solved the task, but required more episodes. Smaller learning rates were slower: `3e-4` eventually solved the environment, while `1e-4` did not reach the target score within 2000 episodes. This suggests that, for this DQN setup, too small a learning rate makes learning too slow, while `1e-3` provides faster convergence without preventing the agent from solving the task.

## Files

Each run's plots, checkpoints, and videos are stored in a separate folder:

- `results/lr_1e-4/`
- `results/lr_3e-4/`
- `results/lr_5e-4/`
- `results/lr_1e-3/`

