from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Hyperparameters aligned with Ha & Schmidhuber (2018); env stays at full resolution."""

    # Raw env observations in .npz are 512×256; bilinear resize keeps 2∶1 aspect (¼ scale → 128×64).
    vae_input_height: int = 64
    vae_input_width: int = 128

    latent_dim: int = 32
    lstm_hidden_dim: int = 256
    mdn_num_gaussians: int = 5

    # Temperature for sampling z_{t+1} from MDN during dream rollouts (paper: τ > 1 often helps).
    dream_temperature: float = 1.25

    # Image scaling for VAE input (match common VAE practice; frames are uint8 in storage).
    obs_scale: float = 1.0 / 255.0
