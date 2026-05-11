"""Roll out episodes in MissileDefense-v0 and save them as `.npz` for V/M training.

Five policy modes are supported, all writing the same `.npz` schema
(obs uint8, action float32, reward float32, done bool):

  * `random`         — uniform random rotation, 10% fire bias (matches the
                      original data-collection convention used for the seed
                      VAE/MDN-RNN training).
  * `controller`     — deterministic policy from a trained controller
                      checkpoint; V is used to encode observations on the fly,
                      M tracks the LSTM hidden state so the controller
                      conditions on (z_t, h_t).
  * `mixed`          — per-step epsilon-greedy: with probability `epsilon` take
                      a uniform-random action (50% fire), else the controller
                      action. Used for iterative data collection: explores
                      around the current policy's trajectory.
  * `teacher`        — hand-coded heuristic policy that targets the most
                      threatening engageable missile and fires when aimed.
                      Reads env internal state directly (no V/M needed).
                      Used to generate kill-event-rich data for the M model
                      under the iter3 HP / burst-fire env redesign.
  * `teacher_mixed`  — teacher + epsilon-greedy: with probability `epsilon`
                      substitute a uniform-random action. Adds exploration
                      around the teacher's near-optimal trajectory so the
                      dataset includes "near-miss" frames and other off-policy
                      states that pure teacher data lacks.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import missile_defense  # noqa: F401 -- registers MissileDefense-v0


def _collect_random(env: gym.Env, fire_bias_keep_prob: float) -> dict:
    """One episode with the legacy random policy. `fire_bias_keep_prob` ∈ [0,1]: with
    that probability force action[1] = -1.0 (don't fire). 0.0 = full uniform random;
    0.9 = the original "fire only 10% of the time" convention.
    """
    obs, info = env.reset()
    observations: list = []
    actions: list = []
    rewards: list = []
    dones: list = []
    done = False
    while not done:
        action = env.action_space.sample()
        if env.np_random.random() < fire_bias_keep_prob:
            action[1] = -1.0
        next_obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        done = terminated or truncated
        dones.append(done)
        obs = next_obs
    return {
        "obs": np.array(observations, dtype=np.uint8),
        "action": np.array(actions, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
    }


def _collect_teacher(env: gym.Env, epsilon: float = 0.0) -> dict:
    """One episode using `ScriptedTeacher` (with optional epsilon-random mixing).

    `epsilon = 0.0` → pure teacher (deterministic given env seed).
    `epsilon > 0.0` → with that probability, substitute a uniform-random action.
    """
    from missile_defense.scripted_teacher import ScriptedTeacher

    teacher = ScriptedTeacher()

    obs, info = env.reset()
    observations: list = []
    actions: list = []
    rewards: list = []
    dones: list = []
    done = False
    while not done:
        if epsilon > 0.0 and env.np_random.random() < epsilon:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = teacher.act(env.unwrapped).astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        done = terminated or truncated
        dones.append(done)
        obs = next_obs
    return {
        "obs": np.array(observations, dtype=np.uint8),
        "action": np.array(actions, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
    }


def _collect_with_world_model(
    env: gym.Env,
    epsilon: float,
    vae,
    mdnrnn,
    ctrl,
    vae_h: int,
    vae_w: int,
    obs_scale: float,
    hidden_d: int,
    device: str,
) -> dict:
    """One episode using controller + V/M; with prob `epsilon` substitute a uniform
    random action at any given step. Pure controller is `epsilon=0.0`.
    """
    import torch  # local import so the random-only mode has no torch dependency

    from world_model.dataset import preprocess_obs_for_vae

    obs, info = env.reset()
    h = torch.zeros(1, hidden_d, device=device)
    c = torch.zeros(1, hidden_d, device=device)
    observations: list = []
    actions: list = []
    rewards: list = []
    dones: list = []
    done = False
    with torch.no_grad():
        while not done:
            if env.np_random.random() < epsilon:
                action = env.action_space.sample()
            else:
                x = preprocess_obs_for_vae(obs, vae_h, vae_w, obs_scale, device=device).unsqueeze(0)
                mu, _ = vae.encode(x)
                a = ctrl(mu, h).squeeze(0).cpu().numpy()
                action = a.astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            done = terminated or truncated
            dones.append(done)

            # Always advance M's hidden state so the controller's next-step decision
            # is conditioned on a coherent trajectory, even on steps where we used a
            # random action. (We feed the action that was actually taken.)
            x_now = preprocess_obs_for_vae(obs, vae_h, vae_w, obs_scale, device=device).unsqueeze(0)
            mu_now, _ = vae.encode(x_now)
            a_t = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
            _, _, _, _, _, h, c = mdnrnn.step_one(mu_now, a_t, h, c)

            obs = next_obs
    return {
        "obs": np.array(observations, dtype=np.uint8),
        "action": np.array(actions, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
    }


def _load_world_model_for_rollouts(
    vae_ckpt: Path, mdnrnn_ckpt: Path, controller_ckpt: Path, device: str
) -> dict:
    """Load V, M, and a Controller initialized from `controller_ckpt`."""
    import torch

    from world_model.config import WorldModelConfig
    from world_model.controller.model import Controller
    from world_model.rnn.model import MDNRNN
    from world_model.vae.model import VAE

    cfg = WorldModelConfig()

    v_ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    if "input_height" in v_ckpt and "input_width" in v_ckpt:
        vae_h, vae_w = int(v_ckpt["input_height"]), int(v_ckpt["input_width"])
    elif "input_size" in v_ckpt:
        s = int(v_ckpt["input_size"])
        vae_h, vae_w = s, s
    else:
        vae_h, vae_w = cfg.vae_input_height, cfg.vae_input_width
    latent_d = int(v_ckpt.get("latent_dim", cfg.latent_dim))
    vae = VAE(latent_dim=latent_d, input_height=vae_h, input_width=vae_w).to(device)
    vae.load_state_dict(v_ckpt["model"])
    vae.eval()

    m_ckpt = torch.load(mdnrnn_ckpt, map_location=device, weights_only=False)
    hidden_d = int(m_ckpt.get("hidden_dim", cfg.lstm_hidden_dim))
    num_g = int(m_ckpt.get("num_gaussians", cfg.mdn_num_gaussians))
    action_d = int(m_ckpt.get("action_dim", 2))
    head_h = int(m_ckpt.get("head_hidden_dim", cfg.head_hidden_dim))
    mdnrnn = MDNRNN(
        latent_dim=latent_d,
        action_dim=action_d,
        hidden_dim=hidden_d,
        num_gaussians=num_g,
        head_hidden_dim=head_h,
    ).to(device)
    # strict=False so older M checkpoints (Linear heads) can still be used for
    # hidden-state evolution during data collection. The reward/done head outputs
    # are discarded here; only h, c are consumed by the controller's next-step
    # action computation.
    result = mdnrnn.load_state_dict(m_ckpt["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"  note: M checkpoint has different head architecture "
            f"({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected). "
            f"Heads are random-init; only LSTM/z_head weights matter for data collection.",
            flush=True,
        )
    mdnrnn.eval()

    ctrl = Controller(latent_dim=latent_d, hidden_dim=hidden_d, action_dim=action_d).to(device)
    c_ckpt = torch.load(controller_ckpt, map_location=device, weights_only=False)
    params = torch.from_numpy(np.array(c_ckpt["params"])).float().to(device)
    ctrl.set_flat_params(params)
    ctrl.eval()

    return {
        "vae": vae,
        "mdnrnn": mdnrnn,
        "ctrl": ctrl,
        "vae_h": vae_h,
        "vae_w": vae_w,
        "obs_scale": cfg.obs_scale,
        "hidden_d": hidden_d,
        "action_d": action_d,
    }


def collect_data(
    num_episodes: int,
    output_dir: str,
    *,
    policy: str = "random",
    epsilon: float = 0.3,
    fire_bias_keep_prob: float = 0.9,
    ep_start: int = 0,
    vae_ckpt: Path = Path("checkpoints/vae/vae.pt"),
    mdnrnn_ckpt: Path = Path("checkpoints/rnn/mdnrnn.pt"),
    controller_ckpt: Path = Path("checkpoints/controller/controller.pt"),
    device: str = "cpu",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    env = gym.make("MissileDefense-v0", render_mode="rgb_array")

    wm = None
    if policy in ("controller", "mixed"):
        wm = _load_world_model_for_rollouts(vae_ckpt, mdnrnn_ckpt, controller_ckpt, device)
        print(
            f"[collect] policy={policy}  epsilon={epsilon}  controller={controller_ckpt}",
            flush=True,
        )
    elif policy == "teacher":
        print(f"[collect] policy=teacher (epsilon=0.0)", flush=True)
    elif policy == "teacher_mixed":
        print(f"[collect] policy=teacher_mixed  epsilon={epsilon}", flush=True)
    else:
        print(
            f"[collect] policy=random  fire_bias_keep_prob={fire_bias_keep_prob}",
            flush=True,
        )

    total_steps = 0
    total_hits = 0
    total_kills = 0
    total_terms = 0
    start_time = time.time()

    for ep in range(num_episodes):
        if policy == "random":
            ep_data = _collect_random(env, fire_bias_keep_prob=fire_bias_keep_prob)
        elif policy == "teacher":
            ep_data = _collect_teacher(env, epsilon=0.0)
        elif policy == "teacher_mixed":
            ep_data = _collect_teacher(env, epsilon=epsilon)
        elif policy == "controller":
            ep_data = _collect_with_world_model(
                env, epsilon=0.0,
                **{k: wm[k] for k in ("vae", "mdnrnn", "ctrl", "vae_h", "vae_w", "obs_scale", "hidden_d")},
                device=device,
            )
        elif policy == "mixed":
            ep_data = _collect_with_world_model(
                env, epsilon=epsilon,
                **{k: wm[k] for k in ("vae", "mdnrnn", "ctrl", "vae_h", "vae_w", "obs_scale", "hidden_d")},
                device=device,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")

        # Under iter3 reward semantics (hit_reward=1.0, kill_reward=2.0):
        #   - Per-step base: step_penalty + fire_penalty = -0.06 (fire steps)
        #     or step_penalty = -0.01 (non-fire).
        #   - Hit step (non-final):   -0.06 + 1.0       = +0.94
        #   - Kill step (final hit):  -0.06 + 1.0 + 2.0 = +2.94
        #   - Terminal protected hit:                    -10.0 (overrides anything)
        # Threshold 2.5 separates kills cleanly; threshold 0.5 captures any laser hit.
        n_hits_ep = int((ep_data["reward"] > 0.5).sum())
        n_kills_ep = int((ep_data["reward"] > 2.5).sum())
        terminated_ep = bool(ep_data["done"][-1] and ep_data["reward"][-1] < -5.0)
        total_hits += n_hits_ep
        total_kills += n_kills_ep
        total_terms += int(terminated_ep)

        out_idx = ep_start + ep
        np.savez_compressed(
            os.path.join(output_dir, f"ep_{out_idx}.npz"),
            obs=ep_data["obs"],
            action=ep_data["action"],
            reward=ep_data["reward"],
            done=ep_data["done"],
        )
        total_steps += int(ep_data["obs"].shape[0])

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - start_time
            fps = total_steps / max(elapsed, 1e-6)
            print(
                f"  ep {ep + 1}/{num_episodes}  total_steps={total_steps}  "
                f"hits/ep={total_hits/(ep+1):.2f}  kills/ep={total_kills/(ep+1):.2f}  "
                f"terms={total_terms}/{ep+1}  fps={fps:.0f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    fps = total_steps / max(elapsed, 1e-6)
    print(
        f"[collect] done. episodes={num_episodes}  total_steps={total_steps}  "
        f"hits/ep_avg={total_hits/num_episodes:.2f}  "
        f"kills/ep_avg={total_kills/num_episodes:.2f}  "
        f"terminated_eps={total_terms}/{num_episodes}  "
        f"FPS={fps:.1f}  ({elapsed:.1f}s)"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/random_rollouts")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "controller", "mixed", "teacher", "teacher_mixed"],
        default="random",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="(mixed policy) probability of taking a uniform-random action at each step.",
    )
    parser.add_argument(
        "--fire-bias-keep-prob",
        type=float,
        default=0.9,
        help="(random policy) prob of overriding action[1] = -1 each step. 0.9 matches the original data convention.",
    )
    parser.add_argument(
        "--ep-start",
        type=int,
        default=0,
        help="Start numbering episodes at this index (so iterative runs don't overwrite).",
    )
    parser.add_argument("--vae-ckpt", type=Path, default=Path("checkpoints/vae/vae.pt"))
    parser.add_argument("--mdnrnn-ckpt", type=Path, default=Path("checkpoints/rnn/mdnrnn.pt"))
    parser.add_argument(
        "--controller-ckpt",
        type=Path,
        default=Path("checkpoints/controller/controller.pt"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    collect_data(
        num_episodes=args.episodes,
        output_dir=args.out,
        policy=args.policy,
        epsilon=args.epsilon,
        fire_bias_keep_prob=args.fire_bias_keep_prob,
        ep_start=args.ep_start,
        vae_ckpt=args.vae_ckpt,
        mdnrnn_ckpt=args.mdnrnn_ckpt,
        controller_ckpt=args.controller_ckpt,
        device=args.device,
    )
