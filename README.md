# Missile Defense Environment

A Gymnasium environment for a 2D Iron Dome-style missile defense game. This environment serves as the simulator for a World Models implementation.

## Overview

A ground turret fires a hitscan laser to intercept incoming ballistic missiles. The agent observes a 512x256 pixel semi-circular radar view (the "radar"). The goal is to survive as long as possible by intercepting missiles before they hit the ground within the protected zone.

## Installation

Ensure you have Python 3.11+ and install the required dependencies:

```bash
pip install gymnasium numpy pygame
```

For **World Models** training (`world_model/`), also install PyTorch:

```bash
pip install -r requirements-training.txt
```

## Running the Environment

### Human Debug View

To play or watch the environment with a full 400x400 world view, trajectories, and the radar boundary overlaid:

```bash
python -m missile_defense.test_env
```

### Data Collection

To collect random policy rollouts for training the VAE (V model):

```bash
python -m missile_defense.collect_data --episodes 100 --out data/random_rollouts
```

This will save compressed `.npz` files in the `data/random_rollouts` directory. Each file contains:
- `obs`: The 512x256 RGB images (shape: `[steps, 256, 512, 3]`, dtype: `uint8`)
- `action`: The continuous actions taken (shape: `[steps, 2]`, dtype: `float32`)
- `reward`: The rewards received (shape: `[steps]`, dtype: `float32`)
- `done`: Whether the episode ended (shape: `[steps]`, dtype: `bool`)

### Extracting Sample Images

If you want to view the actual 512x256 pixel observations that the agent sees, you can extract frames from the collected data:

```bash
python extract_images.py
```

This will save `.png` images to the `data/sample_images/` directory.

## World Models (VAE → MDN-RNN → Controller)

Rollouts stay at **512×256** in `.npz`. The training dataloaders **resize bilinearly** to **128×64** by default (same **2∶1** aspect as the env) before the VAE. See **`world_model/PLAN.md`** for hyperparameters (e.g. **z=32**, **K=5**, **τ**, LSTM **256**), the **dream loop**, and the **reward head** on **M**.

Train the vision model (writes `checkpoints/vae.pt` by default):

```bash
python -m world_model.vae.train --data data/random_rollouts --epochs 20 --ckpt-out checkpoints/vae.pt
```

Train the memory model on latent sequences (requires a VAE checkpoint):

```bash
python -m world_model.rnn.train --data data/random_rollouts --vae-ckpt checkpoints/vae.pt
```

Controller CMA-ES + full imagined rollouts: **`world_model/controller/train.py`** (stub until wired to dreams).

## Testing

To run the physics and environment sanity checks:

```bash
python -m missile_defense.test_physics
```

## Environment Details

- **Action Space**: `Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)`
  - `a[0]`: Turret rotation
  - `a[1]`: Fire laser (fires when `> 0`)
- **Observation Space**: `Box(low=0, high=255, shape=(256, 512, 3), dtype=np.uint8)`
- **Rewards**:
  - `+1.0`: Missile destroyed
  - `-10.0`: Missile impacts the Protected Zone (Episode ends)
  - `0.0`: Missile impacts outside the Protected Zone (Agent successfully ignored a non-threat)
  - `-0.01`: Per timestep (urgency)
  - `-0.05`: Per shot fired (conservation)
