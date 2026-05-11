"""Train MDN-RNN on encoded latents (VAE frozen). See `world_model/PLAN.md`.

Key performance choice: we pre-encode every frame's `z = mu(VAE.encode(obs))` ONCE
at startup. The VAE is frozen so this result is invariant across epochs; recomputing
it inside the per-batch loop was previously the dominant cost. After pre-encoding we
also drop the obs pixel tensors entirely and train the MDN on a small in-RAM tensor
of shape [N_episodes, T_max, latent_dim].
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from world_model.config import WorldModelConfig
from world_model.dataset import RolloutSequenceDataset
from world_model.rnn.model import MDNRNN
from world_model.vae.model import VAE


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    *,
    latent_dim: int,
    action_dim: int,
    hidden_dim: int,
    num_gaussians: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "num_gaussians": num_gaussians,
        },
        path,
    )


def _maybe_resume(
    resume_path: Optional[Path],
    model: MDNRNN,
    opt: torch.optim.Optimizer,
    device: str,
) -> int:
    """Load model/optimizer/epoch from `resume_path`; return the next epoch index."""
    if resume_path is None:
        return 0
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"resumed from {resume_path} at epoch {start_epoch}", flush=True)
    return start_epoch


def _precompute_latents(
    ds: RolloutSequenceDataset,
    vae: VAE,
    device: str,
    encode_batch: int = 512,
) -> Dict[str, torch.Tensor]:
    """Encode every frame's `mu` once, pad all sequences to a common length.

    Returns a dict of CPU tensors:
      z:       [N, T_max, latent_dim] float32
      action:  [N, T_max, action_dim] float32
      reward:  [N, T_max] float32
      mask:    [N, T_max] float32
    """
    n_eps = len(ds)
    latent_d = int(vae.latent_dim)
    action_dim = int(ds[0]["action"].shape[-1])
    lengths = [int(ds[i]["action"].shape[0]) for i in range(n_eps)]
    t_max = max(lengths)

    z_all = torch.zeros(n_eps, t_max, latent_d, dtype=torch.float32)
    a_all = torch.zeros(n_eps, t_max, action_dim, dtype=torch.float32)
    r_all = torch.zeros(n_eps, t_max, dtype=torch.float32)
    m_all = torch.zeros(n_eps, t_max, dtype=torch.float32)

    vae.eval()
    with torch.no_grad():
        for i in range(n_eps):
            ep = ds[i]
            obs = ep["obs"]  # [T, 3, h, w] float32 on CPU
            t_ep = obs.shape[0]
            # Chunk obs through the encoder; full episode can be a few hundred MB on the GPU.
            mus = []
            for start in range(0, t_ep, encode_batch):
                end = min(start + encode_batch, t_ep)
                chunk = obs[start:end].to(device, non_blocking=True)
                mu, _ = vae.encode(chunk)
                mus.append(mu.cpu())
            z_ep = torch.cat(mus, dim=0)
            z_all[i, :t_ep] = z_ep
            a_all[i, :t_ep] = ep["action"]
            r_all[i, :t_ep] = ep["reward"]
            m_all[i, :t_ep] = 1.0

    return {"z": z_all, "action": a_all, "reward": r_all, "mask": m_all}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/random_rollouts"))
    parser.add_argument("--vae-ckpt", type=Path, default=Path("checkpoints/vae/vae.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/rnn/mdnrnn.pt"))
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save an `<ckpt-out-stem>_epoch_NNN.pt` snapshot every N epochs; 0 to disable.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    cfg = WorldModelConfig()

    ckpt = torch.load(args.vae_ckpt, map_location=args.device, weights_only=False)
    if "input_height" in ckpt and "input_width" in ckpt:
        vae_h, vae_w = int(ckpt["input_height"]), int(ckpt["input_width"])
    elif "input_size" in ckpt:
        s = int(ckpt["input_size"])
        vae_h, vae_w = s, s
    else:
        vae_h, vae_w = cfg.vae_input_height, cfg.vae_input_width
    latent_d = int(ckpt.get("latent_dim", cfg.latent_dim))

    print(f"loading dataset from {args.data} ...", flush=True)
    t0 = time.time()
    ds = RolloutSequenceDataset(
        args.data,
        vae_input_height=vae_h,
        vae_input_width=vae_w,
        obs_scale=cfg.obs_scale,
    )
    print(f"  {len(ds)} episodes loaded in {time.time()-t0:.1f}s", flush=True)

    vae = VAE(latent_dim=latent_d, input_height=vae_h, input_width=vae_w).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    print("pre-encoding z for all episodes (frozen VAE, one pass) ...", flush=True)
    t0 = time.time()
    cache = _precompute_latents(ds, vae, args.device)
    n_eps, t_max, _ = cache["z"].shape
    valid_steps = int(cache["mask"].sum().item())
    print(
        f"  done in {time.time()-t0:.1f}s  "
        f"({n_eps} episodes, T_max={t_max}, {valid_steps} valid steps)",
        flush=True,
    )
    # VAE and obs dataset no longer needed; free GPU + CPU memory before training.
    # `ds` holds all decompressed frames at VAE resolution (~7.5 GB for 500 episodes);
    # `cache` is the much smaller latent representation we'll iterate on.
    del vae, ckpt, ds
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    tds = TensorDataset(cache["z"], cache["action"], cache["reward"], cache["mask"])
    loader = DataLoader(tds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    action_dim = int(cache["action"].shape[-1])
    model = MDNRNN(
        latent_dim=latent_d,
        action_dim=action_dim,
        hidden_dim=cfg.lstm_hidden_dim,
        num_gaussians=cfg.mdn_num_gaussians,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = _maybe_resume(args.resume, model, opt, args.device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_mdn = 0.0
        total_r = 0.0
        weight_sum_z = 0.0
        weight_sum_r = 0.0
        ep_start = time.time()
        for z_b, a_b, r_b, m_b in loader:
            z = z_b.to(args.device, non_blocking=True)
            act = a_b.to(args.device, non_blocking=True)
            rew = r_b.to(args.device, non_blocking=True)
            mask = m_b.to(args.device, non_blocking=True)

            pi, mu_z, log_std, r_hat, _, _ = model.forward_unroll(z, act)
            # Output at unroll index k uses (z_k, a_k); h_{k+1} predicts z_{k+1} AND r_k
            # (the reward gained by taking a_k in s_k). target_r is rew[:, :-1], not
            # rew[:, 1:] (which would require knowing a_{k+1} and is non-causal).
            target_z = z[:, 1:]
            target_r = rew[:, :-1].unsqueeze(-1)
            valid_z = mask[:, :-1] * mask[:, 1:]
            valid_r = mask[:, :-1]

            b, t1 = pi.shape[:2]
            flat_pi = pi.reshape(b * t1, latent_d, cfg.mdn_num_gaussians)
            flat_mu = mu_z.reshape(b * t1, latent_d, cfg.mdn_num_gaussians)
            flat_ls = log_std.reshape(b * t1, latent_d, cfg.mdn_num_gaussians)
            flat_target = target_z.reshape(b * t1, latent_d)

            nll = MDNRNN.mdn_negative_log_likelihood_per_seq(flat_pi, flat_mu, flat_ls, flat_target)
            w_z = valid_z.reshape(-1)
            denom_z = w_z.sum().clamp_min(1.0)
            loss_mdn = (nll * w_z).sum() / denom_z

            mse = F.mse_loss(r_hat, target_r, reduction="none").squeeze(-1)
            denom_r = valid_r.sum().clamp_min(1.0)
            loss_r = (mse * valid_r).sum() / denom_r

            loss = loss_mdn + loss_r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_mdn += float(loss_mdn.item()) * float(denom_z)
            total_r += float(loss_r.item()) * float(denom_r)
            weight_sum_z += float(denom_z)
            weight_sum_r += float(denom_r)

        if weight_sum_z <= 0:
            continue
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"mdn={total_mdn / weight_sum_z:.5f}  reward={total_r / max(weight_sum_r, 1.0):.5f}  "
            f"({time.time()-ep_start:.1f}s)",
            flush=True,
        )

        # Always update the "latest" pointer for easy downstream use.
        _save_checkpoint(
            args.ckpt_out,
            model,
            opt,
            epoch=epoch + 1,
            latent_dim=latent_d,
            action_dim=action_dim,
            hidden_dim=cfg.lstm_hidden_dim,
            num_gaussians=cfg.mdn_num_gaussians,
        )
        # Snapshot at the requested cadence.
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            snap = args.ckpt_out.with_name(f"{args.ckpt_out.stem}_epoch_{epoch + 1:03d}{args.ckpt_out.suffix}")
            _save_checkpoint(
                snap,
                model,
                opt,
                epoch=epoch + 1,
                latent_dim=latent_d,
                action_dim=action_dim,
                hidden_dim=cfg.lstm_hidden_dim,
                num_gaussians=cfg.mdn_num_gaussians,
            )

    print(f"saved {args.ckpt_out}")


if __name__ == "__main__":
    main()
