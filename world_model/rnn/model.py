"""
MDN-RNN: LSTMCell on [z_t, a_t] with mixture-density head for z_{t+1} and a reward head.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
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

        self.lstm = nn.LSTMCell(latent_dim + action_dim, hidden_dim)
        out_per_dim = num_gaussians * 3
        self.z_head = nn.Linear(hidden_dim, latent_dim * out_per_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def split_z_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pi_logits, mu, log_std each [B, latent_dim, K]."""
        b = h.size(0)
        raw = self.z_head(h).view(b, self.latent_dim, self.num_gaussians, 3)
        pi_logits = raw[..., 0]
        mu = raw[..., 1]
        log_std = raw[..., 2]
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
        Teacher-forced roll for training.

        Args:
          z_seq: [B, T, latent_dim]
          a_seq: [B, T, action_dim]

        Returns:
          pi_logits, mu, log_std for steps 0..T-2 predicting z_{t+1}, each [B, T-1, D, K]
          reward_hat [B, T-1, 1] predicting reward after the transition
          h_last, c_last: final LSTM state after processing step T-2
        """
        b, t, _ = z_seq.shape
        h = torch.zeros(b, self.hidden_dim, device=z_seq.device, dtype=z_seq.dtype) if h0 is None else h0
        c = torch.zeros(b, self.hidden_dim, device=z_seq.device, dtype=z_seq.dtype) if c0 is None else c0

        pi_list, mu_list, ls_list, r_list = [], [], [], []
        for step in range(t - 1):
            inp = torch.cat([z_seq[:, step], a_seq[:, step]], dim=-1)
            h, c = self.lstm(inp, (h, c))
            pi, mu, log_std = self.split_z_params(h)
            pi_list.append(pi)
            mu_list.append(mu)
            ls_list.append(log_std)
            r_list.append(self.reward_head(h))

        def stack_time(x: list) -> torch.Tensor:
            return torch.stack(x, dim=1)

        return stack_time(pi_list), stack_time(mu_list), stack_time(ls_list), stack_time(r_list), h, c

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
