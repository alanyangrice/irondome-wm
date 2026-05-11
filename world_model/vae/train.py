"""Train VAE on rollout frames (downsampled at dataset init, kept in RAM)."""

from __future__ import annotations

import argparse
import time
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
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight in the ELBO")
    args = parser.parse_args()

    cfg = WorldModelConfig()
    print(f"loading dataset from {args.data} ...", flush=True)
    t0 = time.time()
    ds = RolloutFrameDataset(
        args.data,
        vae_input_height=cfg.vae_input_height,
        vae_input_width=cfg.vae_input_width,
        obs_scale=cfg.obs_scale,
    )
    print(f"  {len(ds)} frames at {cfg.vae_input_width}x{cfg.vae_input_height} loaded in {time.time()-t0:.1f}s", flush=True)
    # num_workers=0: frames already in RAM as uint8, IPC/spawn overhead would be pure cost.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = VAE(
        latent_dim=cfg.latent_dim,
        input_height=cfg.vae_input_height,
        input_width=cfg.vae_input_width,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        total_rec = 0.0
        total_kl = 0.0
        count = 0
        ep_start = time.time()
        for batch in loader:
            batch = batch.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl = VAE.loss(recon, batch, mu, logvar, beta=args.beta)
            loss.backward()
            opt.step()
            n = batch.size(0)
            total += loss.item() * n
            total_rec += recon_loss.item() * n
            total_kl += kl.item() * n
            count += n
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"loss={total / max(count, 1):.3f}  recon={total_rec / max(count, 1):.3f}  "
            f"kl={total_kl / max(count, 1):.3f}  ({time.time()-ep_start:.1f}s)",
            flush=True,
        )

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
