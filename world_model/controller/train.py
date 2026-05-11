"""Controller training via CMA-ES on dream rollouts (see `world_model/PLAN.md`).

Status: the CLI surface and checkpoint convention are in place so this trainer
matches the V/M ones. The dream loop + CMA-ES outer optimizer is not yet wired;
calling `main` raises `NotImplementedError` until that work lands.

When implemented, this will:
  1. Load a frozen VAE and a frozen MDN-RNN from their default checkpoints.
  2. Initialize a `Controller(latent_dim, hidden_dim, action_dim)` linear policy.
  3. Run CMA-ES over the flat parameter vector. Each evaluation samples one or
     more dream rollouts using `MDNRNN.step_one` + `MDNRNN.sample_z` and sums
     the predicted reward head outputs.
  4. Save per-generation snapshots to `checkpoints/controller/controller_gen_NNN.pt`
     and overwrite `checkpoints/controller/controller.pt` with the best-so-far
     parameters. `--resume PATH` will restore CMA-ES state (mean, sigma, generation)
     in addition to the controller params.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", type=Path, default=Path("checkpoints/vae/vae.pt"))
    parser.add_argument("--mdnrnn-ckpt", type=Path, default=Path("checkpoints/rnn/mdnrnn.pt"))
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/controller/controller.pt"))
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--rollouts-per-eval", type=int, default=4)
    parser.add_argument("--dream-horizon", type=int, default=200)
    parser.add_argument("--dream-temperature", type=float, default=1.25)
    parser.add_argument("--sigma-init", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Snapshot generation cadence; 0 to disable per-generation saves.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()  # noqa: F841 -- intentionally parsed so --help works.
    raise NotImplementedError(
        "Wire CMA-ES + MDNRNN dream rollouts + cumulative reward-head outputs. "
        "CLI surface is finalized: snapshots will land under "
        "checkpoints/controller/controller_gen_NNN.pt and the latest pointer is "
        "checkpoints/controller/controller.pt."
    )


if __name__ == "__main__":
    main()
