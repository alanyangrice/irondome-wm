from dataclasses import dataclass

@dataclass
class EnvConfig:
    # Physics
    gravity: float = 2.0
    dt: float = 0.1
    terminal_velocity: float = 40.0  # Max downward speed
    
    # Turret
    omega_max: float = 5.0           # Max rotation per step (degrees)
    # Burst-fire mechanic (iter3): laser may fire `burst_limit` frames in a row,
    # then is forced to skip `burst_cooldown` frames before it can fire again.
    # Voluntarily skipping a frame resets the burst counter to 0 (so 3 fires +
    # 1 voluntary skip = back to a full 5-burst). The legacy `cooldown` field is
    # superseded by this mechanism and no longer consulted by the env.
    burst_limit: int = 5             # Max consecutive fires before forced rest
    burst_cooldown: int = 1          # Forced rest frames after hitting burst_limit
    hit_radius: float = 2.5          # Laser hit tolerance
    laser_radius: float = 64.0       # Max range of the laser (4:1 ratio with radar)

    # Missiles
    spawn_interval: int = 60         # Initial steps between missile spawns
    min_spawn_interval: int = 20
    difficulty_ramp_every: int = 10  # Kills before difficulty increase
    max_missiles: int = 8
    missile_v0: float = 65.0         # Initial launch speed
    min_spawn_angle: float = 10.0    # Minimum launch angle (degrees)
    max_spawn_angle: float = 80.0    # Maximum launch angle (degrees)
    # HP system (iter3): missile takes `missile_hp` hits before exploding. Each
    # non-final hit yields `hit_reward`; the final hit additionally yields
    # `kill_reward`. Damaged missiles that reach the ground produce no extra
    # reward (only the existing ground-impact rewards apply), so partial damage
    # without a kill is worthless to the agent.
    missile_hp: int = 3
    
    # Distractors
    max_clouds: int = 20
    max_birds: int = 5
    cloud_min_y: float = 150.0       # Minimum height for clouds (gives reaction time below them)
    cloud_max_y: float = 300.0       # Maximum height for clouds
    
    # World & Observation
    radar_radius: float = 256.0
    protected_zone_width: float = 80.0 # x in [-40, 40]
    # iter3: explicit control of how often spawned missiles target the
    # protected zone. With uniform-over-radar targeting, ~15.6% of missiles
    # would target the protected column (80 / 512). Raising this to 0.25 means
    # ~25% of spawns are genuine threats, which:
    #   - makes the teacher engage more often per episode → denser kill data
    #   - makes M's done head see more terminal events
    #   - keeps the env meaningfully harder than "fire-at-nothing wins"
    protected_zone_target_prob: float = 0.25
    obs_width: int = 512
    obs_height: int = 256
    
    # Episode
    max_steps: int = 3000
    
    # Rewards (shaping). These directly define the controller's objective.
    # Watch sign conventions: penalties are negative, rewards are positive.
    #
    # iter3 reward redesign:
    #   - 1-shot kills replaced by HP=3 system (above). Each non-final laser hit
    #     pays `hit_reward`; the final hit pays `hit_reward + kill_reward`.
    #   - With HP=3 / hit_reward=1.0 / kill_reward=2.0, a clean full kill totals
    #     +5.0 (matches the iter2 single-shot kill value), but the reward is now
    #     spread across 3 visible events. This densifies the signal the M reward
    #     head must learn: instead of one 0.14%-frequency +5 spike, M sees
    #     well-aimed-laser-frame events at roughly 3× the rate, all with the
    #     same shaped sign.
    #   - Damaged-but-escaped missiles produce no extra reward / penalty: only
    #     the existing ground-impact rewards apply on impact.
    step_penalty: float = -0.01           # Per-timestep urgency pressure.
    fire_penalty: float = -0.05           # Per-shot conservation pressure.
    hit_reward: float = 1.0               # Per laser hit on a missile (non-final).
    kill_reward: float = 2.0              # Bonus on the final hit that explodes the missile.
    protected_zone_penalty: float = -10.0 # Terminal penalty when a missile hits the city.
    non_protected_impact_reward: float = 0.0  # When a missile lands outside the city zone.
    
    # Rendering
    render_fps: int = 30

DEFAULT_CONFIG = EnvConfig()
