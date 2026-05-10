from dataclasses import dataclass

@dataclass
class EnvConfig:
    # Physics
    gravity: float = 2.0
    dt: float = 0.1
    terminal_velocity: float = 40.0  # Max downward speed
    
    # Turret
    omega_max: float = 5.0           # Max rotation per step (degrees)
    cooldown: int = 2                # Steps between shots (reduced for smaller laser range)
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
    
    # Distractors
    max_clouds: int = 20
    max_birds: int = 5
    cloud_min_y: float = 150.0       # Minimum height for clouds (gives reaction time below them)
    cloud_max_y: float = 300.0       # Maximum height for clouds
    
    # World & Observation
    radar_radius: float = 256.0
    protected_zone_width: float = 80.0 # x in [-40, 40]
    obs_width: int = 512
    obs_height: int = 256
    
    # Episode
    max_steps: int = 3000
    
    # Rendering
    render_fps: int = 30

DEFAULT_CONFIG = EnvConfig()
