"""Hand-coded teacher policy for `MissileDefense-v0`.

The teacher reads the env's internal state directly (missile list and turret
state), selects the most-threatening *engageable* missile, and produces an
action `[rotation, fire]` in the env's action space.

Purpose
-------
This is a data-collection policy, not a learned policy. Its job is to generate
trajectories that are *rich in kill events* so the world model's reward / done
heads have a learnable signal. Random data has ~0.07 kills/episode; this
teacher should get several per episode, producing 2 orders of magnitude more
positive-reward steps in the training data.

Design constraints (see chat log for full reasoning)
----------------------------------------------------
1. Threat filter — only engage missiles whose predicted landing-x falls
   inside (or near) the protected zone. ~84% of spawns target outside the
   zone and are not worth a shot, since `fire_penalty < 0` and
   `non_protected_impact_reward = 0`.
2. Range gating — only attempt to aim at missiles within `~1.5 ×
   laser_radius` (so we start rotating before the missile is in range), and
   only fire when the chosen target is actually inside `laser_radius`.
3. Range-adjusted aim tolerance — `hit_radius` is 2.5 units, which at the
   maximum laser range of 64 corresponds to ~2.24° of angular tolerance.
   At closer range, the tolerance is more forgiving. The teacher computes
   `atan(hit_radius / current_range)` per frame.
4. No lead compensation — the laser is instantaneous, so we just aim at the
   missile's current position every frame. The turret's `omega_max` rotation
   rate is the main limiter.

The teacher does NOT need to track cooldown or burst state explicitly: the env
silently no-ops a fire command during forced cooldown (and no `fire_penalty`
is charged in that case), so issuing `action_fire = +1` whenever we're aimed
at a valid target is correct behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Threat:
    """Lightweight view of a missile for threat-evaluation."""
    missile: object  # missile_defense.entities.Missile
    range_to_turret: float
    aim_angle_deg: float        # angle (in env's degree convention) the turret needs
    aim_tol_deg: float          # range-adjusted angular tolerance for a hit
    time_to_ground: float       # rough ETA (steps) until y reaches 0
    predicted_landing_x: float  # ballistic prediction of landing x
    in_protected_zone: bool     # True if landing inside the protected column


class ScriptedTeacher:
    """Heuristic policy that tracks the most-threatening engageable missile
    and fires when aimed.

    The teacher is *stateless* across calls (it re-evaluates every frame), and
    so does not need an `__init__` from env. Just instantiate and call
    `act(env_unwrapped)` each step.
    """

    def __init__(
        self,
        *,
        engagement_range_mult: float = 2.5,
        protected_margin: float = 5.0,
        idle_angle_deg: float = 90.0,
        min_aim_tol_deg: float = 1.5,
        hold_angle_on_idle: bool = True,
    ) -> None:
        """
        Args:
            engagement_range_mult: aim at a missile when it's within
                `engagement_range_mult * laser_radius`. >1 means we start
                rotating toward it before it's actually shootable. 2.5 gives
                ~40 steps of engagement window after a missile crosses
                160 units, vs ~24 steps at 1.5×; that pre-rotation budget
                is critical because rotation is only 5°/step.
            protected_margin: a missile is considered a "threat" if its
                predicted landing-x is within `protected_zone_width/2 +
                protected_margin`. Margin lets us engage missiles that *might*
                drift into the zone (rough ballistic prediction).
            idle_angle_deg: angle to hold the turret at when no threats are
                visible AND `hold_angle_on_idle=False`. 90° = straight up.
            min_aim_tol_deg: floor on the aim-tolerance computation so we
                don't demand sub-degree precision at very long range.
            hold_angle_on_idle: if True, when no threats are visible the
                teacher holds the turret at its current angle rather than
                rotating back to `idle_angle_deg`. Most successive threats
                arrive on similar sides, so holding the previous engagement
                angle gives a head-start on the next.
        """
        self.engagement_range_mult = float(engagement_range_mult)
        self.protected_margin = float(protected_margin)
        self.idle_angle_deg = float(idle_angle_deg)
        self.min_aim_tol_deg = float(min_aim_tol_deg)
        self.hold_angle_on_idle = bool(hold_angle_on_idle)

    # ------------------------------------------------------------------ #
    # Threat evaluation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ballistic_landing_x(m, config) -> float:
        """Predict where missile `m` will land (y=0) by analytically
        integrating its current state under gravity. Ignores terminal-velocity
        clamp for simplicity (the bias is small and only matters near
        terminal speed, which doesn't change which missiles cross the
        protected zone much).

        Solves: y0 + vy0 * t - 0.5 * g_per_step^2 * t * (t-1)  ≈ 0
        but for a rough heuristic we use continuous-time kinematics:
            y(τ) = y0 + vy0 * τ - 0.5 * g * τ^2
        with τ in physical seconds (= step * dt). We return x(τ_land).
        """
        y0 = m.y
        vy0 = m.vy
        g = config.gravity
        # If missile is rising (vy > 0), we still want the time to come back
        # down to y=0. Solve y0 + vy0*τ - 0.5*g*τ^2 = 0 → quadratic.
        # Use the larger positive root (descending crossing of y=0).
        a = -0.5 * g
        b = vy0
        c = y0
        disc = b * b - 4 * a * c
        if disc < 0:
            # Numerical edge case; just fall back to "instant landing".
            return m.x
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        # Take the larger positive root: that's when y(τ) returns to 0 from
        # above (since the parabola opens down: a < 0).
        candidates = [t for t in (t1, t2) if t > 0]
        if not candidates:
            return m.x
        t_land = max(candidates)
        return m.x + m.vx * t_land

    @staticmethod
    def _aim_angle_for_target(turret_x: float, turret_y: float, target_x: float, target_y: float) -> float:
        """Returns the turret angle (in degrees) needed to aim at the target.

        Turret angle convention (from physics.py / turret.py):
            angle = 0   → barrel points along +x  (cos 0 = 1, sin 0 = 0)
            angle = 90  → straight up (cos 90 = 0, sin 90 = 1, and y grows upward)
            angle = 180 → -x
        i.e. `dx = cos(angle)`, `dy = sin(angle)` and the world has +y = up.
        So we want `atan2(target_y - turret_y, target_x - turret_x)`.
        """
        rad = math.atan2(target_y - turret_y, target_x - turret_x)
        deg = math.degrees(rad)
        # Clamp to [0, 180] since the turret can't aim into the ground.
        return float(max(0.0, min(180.0, deg)))

    @staticmethod
    def _next_frame_state(m, config) -> tuple[float, float]:
        """Predict (x, y) of missile `m` at the END of the next physics step.

        env.step ordering (see env.py):
          1. Turret rotates with this action.
          2. Missiles move one step (gravity then position update).
          3. Laser hit check runs against post-move missile positions.

        So our action's laser fires against `next_frame_state`, NOT the missile's
        current state. Aiming at the current state would always be one step
        behind, which at range 64 is ~3.6° of drift (larger than the ~2.2°
        angular tolerance). This lead compensation is critical.
        """
        vy_next = m.vy - config.gravity * config.dt
        if vy_next < -config.terminal_velocity:
            vy_next = -config.terminal_velocity
        x_next = m.x + m.vx * config.dt
        y_next = m.y + vy_next * config.dt
        return x_next, y_next

    def _evaluate_threats(self, env) -> list[_Threat]:
        """Compute a `_Threat` record for each alive missile that will be in
        `engagement_range_mult * laser_radius` on the NEXT frame. Missiles
        outside the engagement range are skipped (rotating toward them is
        wasteful when there's no other engageable target)."""
        config = env.config
        turret_x, turret_y = env.turret.pos
        engagement_range = self.engagement_range_mult * config.laser_radius
        zone_half = config.protected_zone_width / 2.0 + self.protected_margin

        out: list[_Threat] = []
        for m in env.missiles:
            if not m.alive:
                continue

            # Aim at the missile's NEXT-frame position (lead compensation).
            x_next, y_next = self._next_frame_state(m, config)
            dx = x_next - turret_x
            dy = y_next - turret_y
            r_next = math.hypot(dx, dy)
            if r_next > engagement_range:
                continue

            aim_deg = self._aim_angle_for_target(turret_x, turret_y, x_next, y_next)

            # Angular tolerance: missile must be within hit_radius of the laser
            # ray. At next-frame range r_next, that's atan(hit_radius / r_next).
            # Floor at `min_aim_tol_deg` so we don't demand sub-degree aim.
            aim_tol = math.degrees(math.atan2(config.hit_radius, max(r_next, 1e-6)))
            aim_tol = max(aim_tol, self.min_aim_tol_deg)

            # Rough time-to-ground (in steps). With dt = config.dt seconds per
            # step, and y/|vy| being the physical time, we convert.
            if m.vy < -1e-6:
                t_ground_sec = m.y / (-m.vy)
            else:
                # Rising or hovering: gravity will bring it down eventually.
                t_ground_sec = (m.y + abs(m.vy) * 5.0) / max(config.gravity, 1e-6)

            t_ground_steps = max(t_ground_sec / max(config.dt, 1e-6), 0.0)

            landing_x = self._ballistic_landing_x(m, config)
            in_zone = abs(landing_x) <= zone_half

            out.append(_Threat(
                missile=m,
                range_to_turret=r_next,
                aim_angle_deg=aim_deg,
                aim_tol_deg=aim_tol,
                time_to_ground=t_ground_steps,
                predicted_landing_x=landing_x,
                in_protected_zone=in_zone,
            ))
        return out

    # ------------------------------------------------------------------ #
    # Target selection + action
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_target(threats: list[_Threat]) -> Optional[_Threat]:
        """Pick the missile to engage from the list of in-range threats.

        Priority order:
          1. Missiles that will land in the protected zone (real threats).
             Among those, the one closest to impact (lowest time_to_ground).
          2. If no protected-zone threats are in range, return None and let
             the turret idle. Engaging non-threats costs fire_penalty for
             zero positive reward.
        """
        zone_threats = [t for t in threats if t.in_protected_zone]
        if not zone_threats:
            return None
        # Imminent first.
        zone_threats.sort(key=lambda t: t.time_to_ground)
        return zone_threats[0]

    def act(self, env) -> np.ndarray:
        """Compute one action `[rotation, fire]` for env state.

        `env` is the unwrapped `MissileDefenseEnv` (use `env.unwrapped` if you
        only have the gym-wrapped one).
        """
        config = env.config
        turret = env.turret

        threats = self._evaluate_threats(env)
        target = self._select_target(threats)

        # Idle behavior: by default, hold the current angle (don't rotate).
        # Successive threats typically arrive on similar sides, so the previous
        # engagement angle is a better-than-uniform prior on the next aim.
        if target is None:
            if self.hold_angle_on_idle:
                return np.array([0.0, -1.0], dtype=np.float32)
            angle_err = self.idle_angle_deg - turret.angle
            rot = float(np.clip(angle_err / config.omega_max, -1.0, 1.0))
            return np.array([rot, -1.0], dtype=np.float32)

        # Engagement: aim at target's next-frame position. The fire check must
        # account for the rotation that this same action will apply (env.step
        # rotates the turret *before* the laser-hit check, see env.py).
        angle_err = target.aim_angle_deg - turret.angle  # error from CURRENT angle
        rot = float(np.clip(angle_err / config.omega_max, -1.0, 1.0))

        # Post-rotation angle error = remaining error after applying this action.
        # If |angle_err| <= omega_max we'll reach the desired angle exactly;
        # otherwise we land `omega_max` away from current (still |angle_err| -
        # omega_max from target).
        applied_rot_deg = rot * config.omega_max
        post_rot_err = abs(angle_err - applied_rot_deg)

        in_range = target.range_to_turret <= config.laser_radius
        aimed = post_rot_err <= target.aim_tol_deg
        fire = 1.0 if (in_range and aimed) else -1.0

        return np.array([rot, fire], dtype=np.float32)
