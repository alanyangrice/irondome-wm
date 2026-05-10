import math
import pygame
from collections import deque
from .entity import Entity

class Missile(Entity):
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
        
    def draw(self, surface, world_to_pixel_fn, turret_px):
        if not self.alive:
            return
            
        # Draw trail
        if len(self.trail) > 1:
            pts = [world_to_pixel_fn(tx, ty, turret_px) for tx, ty in self.trail]
            for i in range(len(pts) - 1):
                alpha = (i + 1) / len(pts)
                color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))
                pygame.draw.line(surface, color, pts[i], pts[i+1], 3)
        
        # Draw missile core
        px, py = world_to_pixel_fn(self.x, self.y, turret_px)
        pygame.draw.circle(surface, (255, 100, 50), (px, py), 4)
        pygame.draw.circle(surface, (255, 255, 200), (px, py), 2)

def spawn_missile(config, np_random) -> Missile:
    """
    Spawns a missile from off-screen, targeted at the radar ground span.
    """
    # Launch x: either [-1500, -1000] or [1000, 1500]
    # This ensures they spawn way back
    if np_random.random() < 0.5:
        x_start = np_random.uniform(-1500, -1000)
    else:
        x_start = np_random.uniform(1000, 1500)
        
    y_start = 0.0
    
    # Target x: [-256, 256]
    x_target = np_random.uniform(-256, 256)
    
    # Calculate required angle for this range given fixed v0
    R = abs(x_target - x_start)
    v0 = config.missile_v0
    g = config.gravity
    
    # R = (v0^2 * sin(2*theta)) / g
    sin_2theta = (R * g) / (v0 ** 2)
    
    if sin_2theta > 1.0:
        # Fallback: if unreachable (target too far for v0), shoot at 45 degrees for max range
        theta = math.pi / 4
    else:
        # Two possible angles: low trajectory or high trajectory
        theta1 = 0.5 * math.asin(sin_2theta)
        theta2 = math.pi / 2 - theta1
        # Randomly choose between the high arc and low arc
        theta = theta2 if np_random.random() < 0.5 else theta1

    # Direction
    if x_target < x_start:
        vx = -v0 * math.cos(theta)
    else:
        vx = v0 * math.cos(theta)
        
    vy = v0 * math.sin(theta)
    
    return Missile(x_start, y_start, vx, vy)
