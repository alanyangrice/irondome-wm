"""Quick sanity-evaluation of the ScriptedTeacher policy in the real env.

Runs N episodes, prints per-episode (return, length, hits, kills, terminated)
and a summary table at the end. Used to verify the teacher actually achieves
positive returns (and gets a meaningful number of kills) before committing to
a full data collection run.

Usage:
    python -m missile_defense.eval_teacher --episodes 10 [--epsilon 0.0]
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

import missile_defense  # noqa: F401 -- registers MissileDefense-v0
from missile_defense.scripted_teacher import ScriptedTeacher


def run_one(env: gym.Env, teacher: ScriptedTeacher, epsilon: float = 0.0) -> dict:
    obs, _ = env.reset()
    teacher_steps = 0
    eps_random_steps = 0
    total_reward = 0.0
    rewards: list[float] = []
    done = False
    while not done:
        if epsilon > 0.0 and env.np_random.random() < epsilon:
            action = env.action_space.sample().astype(np.float32)
            eps_random_steps += 1
        else:
            action = teacher.act(env.unwrapped).astype(np.float32)
            teacher_steps += 1
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
        "fires": int((rewards_np < 0.0).sum()),  # any step with negative reward includes fire
        "terminated": bool(terminated),
        "teacher_steps": teacher_steps,
        "random_steps": eps_random_steps,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Probability of substituting a uniform-random action at each step.",
    )
    args = p.parse_args()

    env = gym.make("MissileDefense-v0", render_mode="rgb_array")
    teacher = ScriptedTeacher()

    rets: list[float] = []
    lens: list[int] = []
    hits_list: list[int] = []
    kills_list: list[int] = []
    terms = 0

    t0 = time.time()
    for ep in range(args.episodes):
        stats = run_one(env, teacher, epsilon=args.epsilon)
        rets.append(stats["return"])
        lens.append(stats["length"])
        hits_list.append(stats["hits"])
        kills_list.append(stats["kills"])
        if stats["terminated"]:
            terms += 1
        print(
            f"  ep {ep + 1:2d}/{args.episodes}  "
            f"ret={stats['return']:+8.2f}  "
            f"len={stats['length']:4d}  "
            f"hits={stats['hits']:3d}  "
            f"kills={stats['kills']:3d}  "
            f"term={stats['terminated']}",
            flush=True,
        )
    dt = time.time() - t0

    print()
    print(f"=== teacher sanity-eval ({args.episodes} eps, epsilon={args.epsilon}) ===")
    print(f"  mean return:     {np.mean(rets):+8.2f}   (std {np.std(rets):.2f})")
    print(f"  mean length:     {np.mean(lens):8.1f}   (max {max(lens)})")
    print(f"  mean hits/ep:    {np.mean(hits_list):8.2f}")
    print(f"  mean kills/ep:   {np.mean(kills_list):8.2f}")
    print(f"  terminated eps:  {terms}/{args.episodes}")
    print(f"  wall time:       {dt:.1f}s  ({args.episodes / dt:.2f} eps/s)")
    env.close()


if __name__ == "__main__":
    main()
