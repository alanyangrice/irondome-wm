# World Models Plan (Ha & Schmidhuber, 2018)

This project keeps the **simulator** at full **512×256** RGB (easy debugging, existing data). The **vision model** trains on **smaller** inputs by **bilinear downsampling** inside the dataloader. Default **128×64** keeps the same **2∶1** aspect ratio as **512×256** (¼ scale on each axis). You can try **64×32** for smaller tensors or a **square** size only if you accept anisotropic scaling.

## Pinned hyperparameters (start here)

| Symbol | Value | Notes |
|--------|-------|--------|
| VAE input | 128×64 (configurable) | Proportional to stored 512×256; see `WorldModelConfig` |
| \(z\) | 32 | Increase to 64 only if reconstructions lose missile cues |
| LSTM hidden | 256 | |
| MDN mixtures \(K\) | 5 | Per latent dimension |
| Dream temperature \(\tau\) | 1.25 (tunable) | Scale mixture stddevs when sampling \(z_{t+1}\); paper found \(\tau > 1\) often improves robustness |
| Controller | \(\tanh(W [z_t, h_t] + b)\) | Trained with CMA-ES |

## Components

### V (VAE)

- **Encoder / decoder:** Plain 4-layer stride-2 conv stack (mirrored deconv), not ResNet, unless you stay at very high VAE resolution.
- **Loss:** Reconstruction (e.g. MSE in \([0,1]\) after scaling) + KL on \(q(z|x)\).

### M (MDN-RNN)

- **Core:** LSTM consuming \([z_t, a_t]\) (concatenated), producing hidden \(h_{t+1}\).
- **MDN head:** For each latent dimension, **\(K=5\)** Gaussians: mixture weights (softmax), means, standard deviations (positive). Predict **\(z_{t+1}\)**.
- **Reward head (important):** A small head on the LSTM output predicts **scalar reward** \(\hat{r}_t\) (or the reward at \(t+1\) depending on convention—stay consistent in training). This matches the paper’s optional but practically necessary setup for **dream fitness**: CMA-ES needs a reward signal inside imagined rollouts.

### C (Controller)

- Linear map from \([z_t, h_t]\) to actions, \(\tanh\) bounded.

## Training stages

1. **Collect rollouts** with any policy; store **full-resolution** frames in `.npz` (see `missile_defense/collect_data.py`).
2. **Train V** on random frames (`RolloutFrameDataset`).
3. **Train M** on sequences: encode real frames to \(z\) with frozen **V**, then teacher-forced MDN + **reward** regression (masked padding via `collate_padded_episodes`).
4. **Train C** with CMA-ES on **dream rollouts** (below).

## Dream loop (CMA-ES fitness)

This must be **exact**; accidentally encoding real frames mid-rollout voids the method.

1. Reset: get **one** real frame \(x_0\) from the environment (or from logged data); encode **\(z_0 = \mu(x_0)\)** (or sample; be consistent). Initialize **\(h_0\)** (e.g. zeros).
2. For \(t = 0 \dots T-1\):
   - **\(a_t = C(z_t, h_t)\)**
   - LSTM + MDN: **sample** **\(z_{t+1} \sim \mathrm{MDN}(z_t, a_t, h_t)\)** with temperature **\(\tau\)** on the mixture scales.
   - Predict reward **\(\hat{r}_t\)** from the reward head (paper-style). Accumulate fitness \(\sum \hat{r}_t\) (or use sampled reward if you stochasticize it—usually deterministic head output is enough).
   - **Do not** call **V** on any new pixels during the dream.
3. CMA-ES uses **total dream return** as fitness (optionally over multiple seeds).

## Experiment knobs

- **\(\tau\)**: sweep upward if controller overfits deterministic dreams.
- **VAE size**: e.g. 64×32 vs 128×64—same stored data, change only `vae_input_*` in config / dataloader.
- **\(z\)**: 32 → 64 if needed; avoid 128 unless diagnostics demand it.

## Optional deviations (documented)

Environment extras (clouds, birds, grid, trails) are **not** from the paper but improve learnability for a sparse radar scene.
