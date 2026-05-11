"""Train MDN-RNN on encoded latents (VAE frozen). See `world_model/PLAN.md`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from world_model.config import WorldModelConfig
from world_model.dataset import RolloutSequenceDataset, collate_padded_episodes
from world_model.rnn.model import MDNRNN
from world_model.vae.model import VAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/random_rollouts"))
    parser.add_argument("--vae-ckpt", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/mdnrnn.pt"))
    args = parser.parse_args()

    cfg = WorldModelConfig()

    ckpt = torch.load(args.vae_ckpt, map_location=args.device)
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
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_padded_episodes,
        num_workers=0,
    )

    vae = VAE(latent_dim=latent_d, input_height=vae_h, input_width=vae_w).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    action_dim = int(ds[0]["action"].shape[-1])
    model = MDNRNN(
        latent_dim=latent_d,
        action_dim=action_dim,
        hidden_dim=cfg.lstm_hidden_dim,
        num_gaussians=cfg.mdn_num_gaussians,
    ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_mdn = 0.0
        total_r = 0.0
        weight_sum_z = 0.0
        weight_sum_r = 0.0
        ep_start = time.time()
        for batch in loader:
            obs = batch["obs"].to(args.device)
            act = batch["action"].to(args.device)
            rew = batch["reward"].to(args.device)
            mask = batch["mask"].to(args.device)

            b, t, c, h, w = obs.shape
            with torch.no_grad():
                flat = obs.reshape(b * t, c, h, w)
                mu, _ = vae.encode(flat)
                z = mu.reshape(b, t, -1)

            pi, mu_z, log_std, r_hat, _, _ = model.forward_unroll(z, act)
            # Output at unroll index k uses (z_k, a_k); h_{k+1} predicts z_{k+1} AND r_k (the
            # reward gained by taking a_k in s_k). target_r must therefore be rew[:, :-1], not
            # rew[:, 1:] (which would require knowing a_{k+1} and is non-causal).
            target_z = z[:, 1:]
            target_r = rew[:, :-1].unsqueeze(-1)
            valid_z = mask[:, :-1] * mask[:, 1:]
            valid_r = mask[:, :-1]

            t1 = pi.size(1)
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

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": latent_d,
            "action_dim": action_dim,
            "hidden_dim": cfg.lstm_hidden_dim,
            "num_gaussians": cfg.mdn_num_gaussians,
        },
        args.ckpt_out,
    )
    print(f"saved {args.ckpt_out}")


if __name__ == "__main__":
    main()
