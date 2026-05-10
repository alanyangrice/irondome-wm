import math
import pygame
from .entity import Entity

class Cloud(Entity):
    def __init__(self, x: float, y: float, vx: float, size: float, alpha: int):
        self.x = x
        self.y = y
        self.vx = vx
        self.size = size
        self.alpha = alpha

    def step(self, config):
        self.x += self.vx * config.dt
        
    def draw(self, surface, world_to_pixel_fn, turret_px):
        px, py = world_to_pixel_fn(self.x, self.y, turret_px)
        pygame.draw.circle(surface, (60, 60, 70), (px, py), int(self.size))
        pygame.draw.circle(surface, (60, 60, 70), (px + int(self.size*0.6), py - int(self.size*0.3)), int(self.size*0.8))
        pygame.draw.circle(surface, (60, 60, 70), (px - int(self.size*0.6), py - int(self.size*0.3)), int(self.size*0.7))
        pygame.draw.circle(surface, (60, 60, 70), (px + int(self.size*1.1), py), int(self.size*0.6))
        pygame.draw.circle(surface, (60, 60, 70), (px - int(self.size*1.1), py), int(self.size*0.5))

def spawn_cloud(config, np_random) -> Cloud:
    # Spawn just outside the radar so they enter quickly
    if np_random.random() < 0.5:
        x = -config.radar_radius - 100
        vx = np_random.uniform(2, 6) # Much slower
    else:
        x = config.radar_radius + 100
        vx = np_random.uniform(-6, -2) # Much slower
        
    y = np_random.uniform(config.cloud_min_y, config.cloud_max_y)
    size = np_random.uniform(40, 100) # Larger clouds
    alpha = 255 # Solid clouds to block radar
    return Cloud(x, y, vx, size, alpha)
