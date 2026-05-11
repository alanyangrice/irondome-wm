"""
Small conv VAE matching the World Models paper spirit (stride-2 CNN, not ResNet).
Supports non-square inputs (e.g. 128×64 from proportional resize of 512×256).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        input_height: int = 64,
        input_width: int = 128,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width

        # Four stride-2 convs: H -> H/16, W -> W/16
        if input_height % 16 != 0 or input_width % 16 != 0:
            raise ValueError(
                f"input_height and input_width must be divisible by 16, got {input_height}x{input_width}"
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_h = input_height // 16
        self.enc_w = input_width // 16
        flat_dim = 256 * self.enc_h * self.enc_w
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, self.enc_h, self.enc_w)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss(
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard per-image ELBO: sum over pixels/latents, mean over batch.

        Returns (total, recon, kl) so callers can log each term. The previous
        formulation averaged over both pixel count (3*H*W) and latent dim, which
        massively over-weighted KL relative to recon and risks posterior collapse.
        """
        # Reconstruction: sum over C*H*W per image, mean over batch.
        recon_per_image = F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1)
        recon_loss = recon_per_image.mean()
        # KL[q(z|x) || N(0, I)]: sum over latents per image, mean over batch.
        kl_per_image = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        kl = kl_per_image.mean()
        return recon_loss + beta * kl, recon_loss, kl
