"""Train VAE on rollout frames (downsampled on the fly)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from world_model.config import WorldModelConfig
from world_model.dataset import RolloutFrameDataset
from world_model.vae.model import VAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/random_rollouts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/vae.pt"))
    args = parser.parse_args()

    cfg = WorldModelConfig()
    ds = RolloutFrameDataset(
        args.data,
        vae_input_height=cfg.vae_input_height,
        vae_input_width=cfg.vae_input_width,
        obs_scale=cfg.obs_scale,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = VAE(
        latent_dim=cfg.latent_dim,
        input_height=cfg.vae_input_height,
        input_width=cfg.vae_input_width,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(args.device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss = VAE.loss(recon, batch, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / max(count, 1):.5f}")

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": cfg.latent_dim,
            "input_height": cfg.vae_input_height,
            "input_width": cfg.vae_input_width,
        },
        args.ckpt_out,
    )
    print(f"saved {args.ckpt_out}")


if __name__ == "__main__":
    main()
