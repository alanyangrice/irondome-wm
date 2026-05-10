from dataclasses import dataclass

@dataclass
class EnvConfig:
    # Physics
    gravity: float = 9.8
    dt: float = 0.1
    terminal_velocity: float = 30.0  # Max downward speed
    
    # Turret
    omega_max: float = 5.0           # Max rotation per step (degrees)
    cooldown: int = 5                # Steps between shots
    hit_radius: float = 2.5          # Laser hit tolerance
    
    # Missiles
    spawn_interval: int = 60         # Initial steps between missile spawns
    min_spawn_interval: int = 20
    difficulty_ramp_every: int = 10  # Kills before difficulty increase
    max_missiles: int = 8
    missile_v0: float = 70.0         # Initial launch speed (increased to reach further targets)
    
    # World & Observation
    radar_radius: float = 128.0
    protected_zone_width: float = 80.0 # x in [-20, 20]
    obs_width: int = 256
    obs_height: int = 128
    
    # Episode
    max_steps: int = 3000
    
    # Rendering
    render_fps: int = 30

DEFAULT_CONFIG = EnvConfig()
