import math
from typing import List
from .entities.turret import Turret
from .entities.missile import Missile

def point_line_distance(px: float, py: float, lx1: float, ly1: float, lx2: float, ly2: float) -> float:
    """Distance from point (px, py) to line segment (lx1, ly1)-(lx2, ly2)"""
    line_mag = math.hypot(lx2 - lx1, ly2 - ly1)
    if line_mag == 0.0:
        return math.hypot(px - lx1, py - ly1)

    u = ((px - lx1) * (lx2 - lx1) + (py - ly1) * (ly2 - ly1)) / (line_mag ** 2)
    
    # We want a ray, not just a segment.
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
            
            # Check if it's within laser range
            dist_to_turret = math.hypot(to_missile_x, to_missile_y)
            
            if dot > 0 and dist_to_turret <= config.laser_radius:
                # It's a valid hit within range. Find the closest one to the turret.
                if dist_to_turret < closest_dist_to_turret:
                    closest_dist_to_turret = dist_to_turret
                    hit_idx = i

    if hit_idx != -1:
        missiles[hit_idx].alive = False
        
    return hit_idx
