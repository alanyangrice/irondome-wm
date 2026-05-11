"""Train VAE on rollout frames (downsampled at dataset init, kept in RAM)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from world_model.config import WorldModelConfig
from world_model.dataset import RolloutFrameDataset
from world_model.vae.model import VAE


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    *,
    latent_dim: int,
    input_height: int,
    input_width: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "latent_dim": latent_dim,
            "input_height": input_height,
            "input_width": input_width,
        },
        path,
    )


def _maybe_resume(
    resume_path: Optional[Path],
    model: VAE,
    opt: torch.optim.Optimizer,
    device: str,
) -> int:
    if resume_path is None:
        return 0
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"resumed from {resume_path} at epoch {start_epoch}", flush=True)
    return start_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/random_rollouts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/vae/vae.pt"))
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight in the ELBO")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save an `<ckpt-out-stem>_epoch_NNN.pt` snapshot every N epochs; 0 to disable.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
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
    print(
        f"  {len(ds)} frames at {cfg.vae_input_width}x{cfg.vae_input_height} loaded in {time.time()-t0:.1f}s",
        flush=True,
    )
    # num_workers=0: frames already in RAM as uint8, IPC/spawn overhead would be pure cost.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = VAE(
        latent_dim=cfg.latent_dim,
        input_height=cfg.vae_input_height,
        input_width=cfg.vae_input_width,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = _maybe_resume(args.resume, model, opt, args.device)

    for epoch in range(start_epoch, args.epochs):
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

        # "latest" pointer (overwritten every epoch).
        _save_checkpoint(
            args.ckpt_out,
            model,
            opt,
            epoch=epoch + 1,
            latent_dim=cfg.latent_dim,
            input_height=cfg.vae_input_height,
            input_width=cfg.vae_input_width,
        )
        # Snapshot at the requested cadence so older states can be restored.
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            snap = args.ckpt_out.with_name(f"{args.ckpt_out.stem}_epoch_{epoch + 1:03d}{args.ckpt_out.suffix}")
            _save_checkpoint(
                snap,
                model,
                opt,
                epoch=epoch + 1,
                latent_dim=cfg.latent_dim,
                input_height=cfg.vae_input_height,
                input_width=cfg.vae_input_width,
            )

    print(f"saved {args.ckpt_out}")


if __name__ == "__main__":
    main()
