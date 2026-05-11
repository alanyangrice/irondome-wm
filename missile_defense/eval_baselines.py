"""Eval random and never-fire baselines in the iter3 env (HP=3 + burst-fire).

Used to anchor the scripted teacher's reported numbers against what trivial
policies achieve under the new mechanics. Old (iter2) baseline numbers are
invalid because the reward structure changed.

Usage: python -m missile_defense.eval_baselines --episodes 10
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

import missile_defense  # noqa: F401


def run_policy(env: gym.Env, policy: str, fire_bias_keep_prob: float = 0.9) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    rewards: list[float] = []
    done = False
    while not done:
        if policy == "never_fire":
            action = np.array([0.0, -1.0], dtype=np.float32)
        elif policy == "random":
            action = env.action_space.sample().astype(np.float32)
            if env.np_random.random() < fire_bias_keep_prob:
                action[1] = -1.0
        else:
            raise ValueError(policy)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        rewards.append(float(reward))
        done = terminated or truncated
    rewards_np = np.asarray(rewards, dtype=np.float32)
    return {
        "return": total_reward,
        "length": int(len(rewards)),
        "hits": int((rewards_np > 0.5).sum()),
        "kills": int((rewards_np > 2.5).sum()),
        "terminated": bool(terminated),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    for policy in ("never_fire", "random"):
        env = gym.make("MissileDefense-v0", render_mode="rgb_array")
        rets, lens, hits, kills, terms = [], [], [], [], 0
        t0 = time.time()
        for _ in range(args.episodes):
            s = run_policy(env, policy)
            rets.append(s["return"])
            lens.append(s["length"])
            hits.append(s["hits"])
            kills.append(s["kills"])
            if s["terminated"]:
                terms += 1
        dt = time.time() - t0
        print(f"=== baseline: {policy} ({args.episodes} eps) ===")
        print(f"  mean return:    {np.mean(rets):+8.2f}   (std {np.std(rets):.2f})")
        print(f"  mean length:    {np.mean(lens):8.1f}")
        print(f"  mean hits/ep:   {np.mean(hits):8.2f}")
        print(f"  mean kills/ep:  {np.mean(kills):8.2f}")
        print(f"  terminated eps: {terms}/{args.episodes}")
        print(f"  wall time:      {dt:.1f}s")
        print()
        env.close()


if __name__ == "__main__":
    main()
