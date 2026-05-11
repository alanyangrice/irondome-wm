"""
MDN-RNN: batched LSTM on [z_t, a_t] with mixture-density head for z_{t+1} and a reward head.

The training path uses cuDNN-fused `nn.LSTM` over the whole padded sequence in a single
kernel launch (vastly faster than `LSTMCell` in a Python `for` loop). The dream-time
autoregressive path calls the same module with `seq_len=1` to keep weights identical.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    # Bounds on the predicted log-stddev. -7 -> std ~9e-4 (tight enough for clustered
    # latents, well away from float underflow). +5 -> std ~148 (much wider than any
    # realistic unit-variance VAE code, but finite to keep the NLL well-defined when
    # an outlier target arrives early in training).
    LOG_STD_MIN: float = -7.0
    LOG_STD_MAX: float = 5.0

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_gaussians: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        out_per_dim = num_gaussians * 3
        self.z_head = nn.Linear(hidden_dim, latent_dim * out_per_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def _format_state(
        self,
        h0: Optional[torch.Tensor],
        c0: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accept (B, H) or (1, B, H) and always return (1, B, H) for nn.LSTM."""
        if h0 is None:
            h0 = torch.zeros(1, batch, self.hidden_dim, device=device, dtype=dtype)
        elif h0.dim() == 2:
            h0 = h0.unsqueeze(0)
        if c0 is None:
            c0 = torch.zeros(1, batch, self.hidden_dim, device=device, dtype=dtype)
        elif c0.dim() == 2:
            c0 = c0.unsqueeze(0)
        return h0, c0

    def split_z_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pi_logits, mu, log_std each [..., latent_dim, K] given last-dim `hidden`.

        Accepts any leading batch shape (e.g. [B, hidden] for one step or
        [B, T, hidden] for whole-sequence prediction).
        """
        raw = self.z_head(h)
        new_shape = raw.shape[:-1] + (self.latent_dim, self.num_gaussians, 3)
        raw = raw.view(*new_shape)
        pi_logits = raw[..., 0]
        mu = raw[..., 1]
        log_std = raw[..., 2].clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return pi_logits, mu, log_std

    @staticmethod
    def mdn_negative_log_likelihood_per_seq(
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        target_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        NLL summed over latent dims per batch row.
        Shapes: pi_logits, mu, log_std are [N, latent_dim, K]; target_z is [N, latent_dim].
        Returns: [N]
        """
        log_pi = F.log_softmax(pi_logits, dim=-1)
        std = torch.exp(log_std).clamp_min(1e-8)
        const = 0.5 * math.log(2 * math.pi)
        diff = (target_z.unsqueeze(-1) - mu) / std
        log_normal = -0.5 * diff.pow(2) - log_std - const
        log_mix = log_pi + log_normal
        log_prob = torch.logsumexp(log_mix, dim=-1)
        return -log_prob.sum(dim=-1)

    @staticmethod
    def mdn_negative_log_likelihood(
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        target_z: torch.Tensor,
    ) -> torch.Tensor:
        """Mean NLL per sequence element (for debugging / unmasked batches)."""
        return MDNRNN.mdn_negative_log_likelihood_per_seq(pi_logits, mu, log_std, target_z).mean()

    def forward_unroll(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forced roll for training, in a single fused LSTM call.

        Args:
          z_seq: [B, T, latent_dim]
          a_seq: [B, T, action_dim]

        Returns:
          pi_logits, mu, log_std for steps 0..T-2 predicting z_{t+1}, each [B, T-1, D, K]
          reward_hat [B, T-1, 1] predicting reward after the transition
          h_last, c_last: final LSTM state, shape [B, hidden] (squeezed)
        """
        b, t, _ = z_seq.shape
        if t < 2:
            raise ValueError(f"need at least 2 timesteps to unroll a transition, got T={t}")
        # Feed inputs at steps 0..T-2; cuDNN handles the full sequence in one call.
        x = torch.cat([z_seq[:, :-1], a_seq[:, :-1]], dim=-1)
        h0_, c0_ = self._format_state(h0, c0, b, z_seq.device, z_seq.dtype)
        out, (h_n, c_n) = self.lstm(x, (h0_, c0_))
        pi_logits, mu, log_std = self.split_z_params(out)
        reward_hat = self.reward_head(out)
        return pi_logits, mu, log_std, reward_hat, h_n.squeeze(0), c_n.squeeze(0)

    def step_one(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step roll for dream-time autoregressive use.

        Args:
          z_t: [B, latent_dim], a_t: [B, action_dim], h/c: [B, hidden] or None.

        Returns the (pi_logits, mu, log_std) for sampling z_{t+1}, the predicted
        reward `r_hat` for the transition, and the new (h, c) each [B, hidden].
        """
        b = z_t.size(0)
        x = torch.cat([z_t, a_t], dim=-1).unsqueeze(1)
        h0_, c0_ = self._format_state(h, c, b, z_t.device, z_t.dtype)
        out, (h_n, c_n) = self.lstm(x, (h0_, c0_))
        out = out.squeeze(1)
        pi_logits, mu, log_std = self.split_z_params(out)
        reward_hat = self.reward_head(out)
        return pi_logits, mu, log_std, reward_hat, h_n.squeeze(0), c_n.squeeze(0)

    @staticmethod
    def sample_z(
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample one z vector per row from the MDN; scale stddevs by `temperature` (τ in the paper).
        Shapes: pi_logits, mu, log_std are [B, D, K].
        """
        pi = F.softmax(pi_logits, dim=-1)
        b, d, _k = pi.shape
        cat = torch.distributions.Categorical(pi)
        idx = cat.sample()
        idx_exp = idx.unsqueeze(-1)
        mu_sel = mu.gather(-1, idx_exp).squeeze(-1)
        log_std_sel = log_std.gather(-1, idx_exp).squeeze(-1)
        std = torch.exp(log_std_sel) * temperature
        eps = torch.randn(b, d, device=mu.device, dtype=mu.dtype)
        return mu_sel + eps * std
