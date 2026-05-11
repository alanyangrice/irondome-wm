"""Visual viewer for the ScriptedTeacher policy.

Opens a pygame window and runs the teacher in human-render mode so the
developer can watch what it's doing: which missile it's locked onto, whether
it's aimed, when it fires, burst-counter state, etc.

Controls:
    [Space]  pause / resume
    [R]      reset and start a new episode
    [Esc] or [Q] / window close  quit

Usage:
    python -m missile_defense.view_teacher [--episodes 5] [--epsilon 0.0]
                                           [--fps 30]
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np
import pygame

import missile_defense  # noqa: F401 -- registers MissileDefense-v0
from missile_defense.scripted_teacher import ScriptedTeacher


def _draw_teacher_overlay(env, teacher: ScriptedTeacher, surface: pygame.Surface) -> None:
    """Draw an annotation on top of the rendered scene showing which missile
    the teacher is engaging this frame, in human-render coordinates.

    The renderer puts the turret at (w//2, h-20) and uses
        px = w//2 + world_x
        py = (h-20) - world_y
    (see Renderer.render_human / world_to_pixel — same mapping with a
    different turret_px).
    """
    w, h = surface.get_size()
    turret_px = (w // 2, h - 20)

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        return (int(turret_px[0] + x), int(turret_px[1] - y))

    threats = teacher._evaluate_threats(env.unwrapped)
    target = teacher._select_target(threats)

    # Engagement-range ring (cyan) around the turret.
    config = env.unwrapped.config
    eng_r = int(teacher.engagement_range_mult * config.laser_radius)
    laser_r = int(config.laser_radius)
    pygame.draw.circle(surface, (50, 150, 200), turret_px, eng_r, 1)
    pygame.draw.circle(surface, (80, 200, 255), turret_px, laser_r, 1)

    # All in-range threats: faint yellow circles.
    for t in threats:
        px, py = world_to_pixel(t.missile.x, t.missile.y)
        color = (200, 200, 60) if t.in_protected_zone else (120, 120, 120)
        pygame.draw.circle(surface, color, (px, py), 8, 1)

    # Selected target: bright red lock-on box + predicted landing-x marker.
    if target is not None:
        m = target.missile
        px, py = world_to_pixel(m.x, m.y)
        pygame.draw.rect(surface, (255, 60, 60), (px - 10, py - 10, 20, 20), 2)

        # Predicted landing position marker on the ground line.
        lx, ly = world_to_pixel(target.predicted_landing_x, 0)
        pygame.draw.line(surface, (255, 60, 60), (lx - 6, ly), (lx + 6, ly), 2)
        pygame.draw.line(surface, (255, 60, 60), (lx, ly - 6), (lx, ly + 6), 2)


def run_episode(env: gym.Env, teacher: ScriptedTeacher, epsilon: float, fps: int) -> tuple[bool, dict]:
    """Run one episode interactively. Returns (quit_requested, stats)."""
    obs, _ = env.reset()
    done = False
    paused = False
    quit_requested = False
    reset_requested = False
    total_reward = 0.0
    rewards: list[float] = []

    # Force one render so the window appears immediately even when paused.
    env.render()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                done = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_requested = True
                    done = True
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_requested = True
                    done = True
        if quit_requested or reset_requested:
            break

        if paused:
            pygame.time.wait(50)
            continue

        if epsilon > 0.0 and env.np_random.random() < epsilon:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = teacher.act(env.unwrapped).astype(np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        rewards.append(float(reward))

        # The env auto-renders in human mode inside step(). After that flip
        # we draw the teacher overlay on top using the renderer's surface.
        renderer = env.unwrapped.renderer
        if renderer.human_surface is not None:
            _draw_teacher_overlay(env.unwrapped, teacher, renderer.human_surface)
            pygame.display.flip()

        renderer.clock.tick(fps)
        done = terminated or truncated

    rewards_np = np.asarray(rewards, dtype=np.float32) if rewards else np.zeros(0)
    stats = {
        "return": float(total_reward),
        "length": int(len(rewards)),
        "hits": int((rewards_np > 0.5).sum()),
        "kills": int((rewards_np > 2.5).sum()),
        "terminated": bool(terminated) if rewards else False,
    }
    return quit_requested, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    env = gym.make("MissileDefense-v0", render_mode="human")
    teacher = ScriptedTeacher()

    pygame.init()
    print("Controls: [Space]=pause, [R]=reset, [Esc/Q]=quit")
    print()

    ep = 0
    while ep < args.episodes:
        ep += 1
        print(f"-- episode {ep}/{args.episodes} starting...")
        quit_requested, stats = run_episode(env, teacher, args.epsilon, args.fps)
        print(
            f"   ret={stats['return']:+8.2f}  len={stats['length']:4d}  "
            f"hits={stats['hits']:3d}  kills={stats['kills']:3d}  "
            f"term={stats['terminated']}"
        )
        if quit_requested:
            print("-- quit requested")
            break

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
