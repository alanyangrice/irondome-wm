import math
import pygame
from .entity import Entity

class Bird(Entity):
    def __init__(self, x: float, y: float, vx: float, vy_amp: float, vy_freq: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy_amp = vy_amp
        self.vy_freq = vy_freq
        self.t = 0.0

    def step(self, config):
        self.x += self.vx * config.dt
        # Use cosine for velocity to make position a sine wave, or just use sine for position directly
        # If we just add to y, we are integrating velocity. 
        # y += sin(t) * amp * dt means y will oscillate.
        self.y += math.sin(self.t * self.vy_freq) * self.vy_amp * config.dt
        self.t += config.dt
        
    def draw(self, surface, world_to_pixel_fn, turret_px):
        px, py = world_to_pixel_fn(self.x, self.y, turret_px)
        wing_offset = int(math.sin(self.t * self.vy_freq * 2) * 5)
        pygame.draw.line(surface, (120, 120, 120), (px, py), (px - 6, py - 3 + wing_offset), 2)
        pygame.draw.line(surface, (120, 120, 120), (px, py), (px + 6, py - 3 + wing_offset), 2)

def spawn_bird(config, np_random) -> Bird:
    if np_random.random() < 0.5:
        x = -config.radar_radius - 100
        vx = np_random.uniform(5, 12) # Slower birds
    else:
        x = config.radar_radius + 100
        vx = np_random.uniform(-12, -5) # Slower birds
        
    y = np_random.uniform(50, config.radar_radius - 50)
    vy_amp = np_random.uniform(2, 6) # Reduced vertical amplitude so they don't bounce too high
    vy_freq = np_random.uniform(2, 4) # Slightly faster flapping frequency
    return Bird(x, y, vx, vy_amp, vy_freq)
