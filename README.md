# TD3 for Competitive Air Hockey

**RL Course WS 2025/26 — Final Project**
*Team: alphabet-td3 · Julian Jurcevic*

A Twin Delayed Deep Deterministic Policy Gradient (TD3) agent trained to play competitive two-player air hockey. The agent uses curriculum learning, Ornstein–Uhlenbeck exploration noise, and self-play to achieve robust performance against both scripted and unseen opponents.

## Results

| Opponent | Win Rate | Avg Return |
|----------|----------|------------|
| Weak     | ~95%     | ~8.5       |
| Strong   | ~85%     | ~7.0       |

## Project Structure

```
├── rl/                          # Core RL framework
│   ├── td3/                     # TD3 implementation
│   │   ├── agent.py             # TD3 agent (action selection, noise, checkpointing)
│   │   ├── learner.py           # Critic/actor updates, soft target updates
│   │   ├── networks.py          # Actor and TwinQ network architectures
│   │   └── config.py            # Hyperparameter dataclass
│   ├── training/
│   │   ├── train.py             # Main training loop
│   │   ├── curricula.py         # Opponent scheduling definitions
│   │   ├── opponent_manager.py  # Opponent selection logic
│   │   └── self_play.py         # Snapshot pool with difficulty-weighted sampling
│   ├── replay/
│   │   ├── base_buffer.py       # Base replay buffer
│   │   ├── uniform_buffer.py    # Uniform sampling
│   │   └── prioritized_buffer.py # PER with importance weighting
│   ├── common/
│   │   ├── noise.py             # Gaussian, OU, Pink, Uniform noise
│   │   ├── scaler.py            # Action/observation scaling
│   │   └── device.py            # CUDA/CPU device selection
│   ├── experiment/
│   │   ├── definitions.py       # Predefined experiment configs (stages, ablations)
│   │   ├── scheduler.py         # Sequential experiment runner
│   │   ├── directories.py       # Run directory management
│   │   └── tracking.py          # Run metadata, seeding, config serialization
│   ├── utils/
│   │   ├── evaluator.py         # Evaluation against fixed opponents
│   │   ├── plotter.py           # Training curve visualization
│   │   ├── metrics.py           # Metric tracking and serialization
│   │   ├── model_manager.py     # Best-model checkpointing
│   │   ├── logger.py            # File/console logging
│   │   └── early_stopping.py    # Patience-based early stopping
│   ├── main.py                  # Entry point for training
│   └── play.py                  # Watch a trained agent play
├── hockey/                      # Hockey environment (local copy)
├── competition/                 # Tournament client
│   └── run_client.py            # CompRL server connection
├── pretrained/                  # Pretrained checkpoints (stage 1–3)
├── latex/
│   ├── report/                  # LaTeX report source
│   └── presentation/            # LaTeX presentation source
└── model_evaluation/            # Plotting scripts for final figures
```

## Setup

```bash
# Clone
git clone https://github.com/julilili42/air-hockey-td3.git
cd air-hockey-td3

# Install dependencies
pip install -e .

# The hockey environment installs automatically via setup.py.
# For the competition client:
pip install -r competition/requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, Gymnasium

## Training

Training is organized into stages via the experiment scheduler. Each stage builds on the previous checkpoint.

```bash
# Stage 1: Train against weak opponent (10k episodes)
python -m rl.main --experiment stage1 --seed 42

# Stage 2: Mixed curriculum — weak + strong (25k episodes)
python -m rl.main --experiment stage2 --seed 42

# Stage 3: Strong + self-play emphasis (20k episodes)
python -m rl.main --experiment stage3 --seed 42
```

Ablation studies:

```bash
# Noise comparison (Gaussian, OU, Pink, Uniform)
python -m rl.main --experiment noise --seed 42

# Self-play & PER ablation (2x2 grid)
python -m rl.main --experiment sp_per --seed 42
```

Outputs (models, metrics, plots) are saved to `rl/cluster_runs/<timestamp>/`.

## Evaluation

Watch a trained agent play against the built-in opponent:

```bash
# Against weak opponent
python -m rl.play --weak

# Against strong opponent
python -m rl.play
```

The model path is configured in `rl/play.py`.

## Report & Presentation

- **Report:** [`latex/report/template.pdf`](latex/report/template.pdf)
- **Presentation:** [`latex/presentation/main.pdf`](latex/presentation/main.pdf)

## Method Summary

**TD3** with three key modifications for the competitive setting:

- **Curriculum Learning:** Three-stage training progressing from weak-only → mixed → strong + self-play, with stage transitions triggered at 85% weak win rate.
- **OU Exploration Noise:** Temporally correlated noise with linear annealing. Outperformed Gaussian (+8% WR against strong), Pink, and Uniform noise.
- **Self-Play:** Policy snapshots stored every 150 episodes in a pool of 25. Opponents sampled via difficulty-weighted scores (×1.2 on loss, ×0.95 on win).

Key hyperparameters: γ=0.99, τ=0.005, LR=2×10⁻⁴, batch=256, buffer=300k, hidden=2×256.

## References

- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* (TD3), ICML 2018
- Eberhard et al., *Pink Noise Is All You Need*, ICLR 2023
- Schaul et al., *Prioritized Experience Replay*, ICLR 2016
- Bengio et al., *Curriculum Learning*, ICML 2009
- Bansal et al., *Emergent Complexity via Multi-Agent Competition*, ICLR 2018
