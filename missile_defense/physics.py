import math
import numpy as np
from collections import deque
from typing import List, Tuple

class Turret:
    def __init__(self, config):
        self.config = config
        self.angle = 90.0  # Pointing straight up
        self.cooldown_timer = 0
        self.pos = (0.0, 0.0)

    def reset(self):
        self.angle = 90.0
        self.cooldown_timer = 0

    def step(self, action_rot: float, action_fire: float) -> bool:
        """
        Updates turret state. Returns True if a laser is fired this step.
        action_rot: [-1, 1]
        action_fire: > 0 to fire
        """
        # Update angle
        delta_angle = action_rot * self.config.omega_max
        self.angle += delta_angle
        self.angle = np.clip(self.angle, 0.0, 180.0)

        # Update cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # Fire laser
        fired = False
        if action_fire > 0 and self.cooldown_timer == 0:
            fired = True
            self.cooldown_timer = self.config.cooldown

        return fired

class Missile:
    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alive = True
        self.trail = deque(maxlen=5)
        self.trail.append((x, y))

    def step(self, config):
        if not self.alive:
            return

        # Gravity
        self.vy -= config.gravity * config.dt
        
        # Terminal velocity (cap downward speed)
        if self.vy < -config.terminal_velocity:
            self.vy = -config.terminal_velocity

        self.x += self.vx * config.dt
        self.y += self.vy * config.dt
        
        self.trail.append((self.x, self.y))

def point_line_distance(px: float, py: float, lx1: float, ly1: float, lx2: float, ly2: float) -> float:
    """Distance from point (px, py) to line segment (lx1, ly1)-(lx2, ly2)"""
    line_mag = math.hypot(lx2 - lx1, ly2 - ly1)
    if line_mag == 0.0:
        return math.hypot(px - lx1, py - ly1)

    u = ((px - lx1) * (lx2 - lx1) + (py - ly1) * (ly2 - ly1)) / (line_mag ** 2)
    
    # We want a ray, not just a segment.
    # Actually, for a ray starting at (lx1, ly1) and going towards (lx2, ly2):
    if u < 0.0:
        # Behind the origin of the ray
        return math.hypot(px - lx1, py - ly1)
    
    # Closest point on the line
    ix = lx1 + u * (lx2 - lx1)
    iy = ly1 + u * (ly2 - ly1)
    return math.hypot(px - ix, py - iy)

def check_laser_hits(turret: Turret, missiles: List[Missile], config) -> int:
    """
    Checks if the laser hits any missiles. 
    Returns the index of the hit missile, or -1 if none.
    """
    # Ray direction
    rad = math.radians(turret.angle)
    dx = math.cos(rad)
    dy = math.sin(rad)
    
    # We use a point far along the ray to define the line segment for distance calculation
    ray_end_x = turret.pos[0] + dx * 1000.0
    ray_end_y = turret.pos[1] + dy * 1000.0

    closest_dist_to_turret = float('inf')
    hit_idx = -1

    for i, m in enumerate(missiles):
        if not m.alive:
            continue
            
        # Distance from missile to ray
        dist_to_ray = point_line_distance(m.x, m.y, turret.pos[0], turret.pos[1], ray_end_x, ray_end_y)
        
        if dist_to_ray <= config.hit_radius:
            # Check if it's in front of the turret (dot product > 0)
            to_missile_x = m.x - turret.pos[0]
            to_missile_y = m.y - turret.pos[1]
            dot = to_missile_x * dx + to_missile_y * dy
            
            # Check if it's within radar range (laser max range)
            dist_to_turret = math.hypot(to_missile_x, to_missile_y)
            
            if dot > 0 and dist_to_turret <= config.radar_radius:
                # It's a valid hit within range. Find the closest one to the turret.
                if dist_to_turret < closest_dist_to_turret:
                    closest_dist_to_turret = dist_to_turret
                    hit_idx = i

    if hit_idx != -1:
        missiles[hit_idx].alive = False
        
    return hit_idx

def spawn_missile(config, np_random) -> Missile:
    """
    Spawns a missile from off-screen, targeted at the radar ground span.
    """
    # Launch x: either [-250, -150] or [150, 250]
    # This ensures they always spawn outside the 128-unit radar radius
    if np_random.random() < 0.5:
        x_start = np_random.uniform(-250, -150)
    else:
        x_start = np_random.uniform(150, 250)
        
    y_start = 0.0
    
    # Target x: [-128, 128]
    x_target = np_random.uniform(-128, 128)
    
    # Calculate required angle for this range
    R = abs(x_target - x_start)
    v0 = config.missile_v0
    g = config.gravity
    
    # R = (v0^2 * sin(2*theta)) / g
    sin_2theta = (R * g) / (v0 ** 2)
    
    if sin_2theta > 1.0:
        # Fallback: if unreachable (e.g., config changed), just shoot at 45 degrees
        theta = math.pi / 4
    else:
        # Two possible angles: low trajectory or high trajectory
        # Let's pick high trajectory for more air time
        theta1 = 0.5 * math.asin(sin_2theta)
        theta2 = math.pi / 2 - theta1
        theta = theta2 if np_random.random() < 0.7 else theta1 # 70% high, 30% low

    # Direction
    if x_target < x_start:
        vx = -v0 * math.cos(theta)
    else:
        vx = v0 * math.cos(theta)
        
    vy = v0 * math.sin(theta)
    
    return Missile(x_start, y_start, vx, vy)
