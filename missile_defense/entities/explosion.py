import pygame
from .entity import Entity

class Explosion(Entity):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.age = 0
        self.alive = True
        
    def step(self, config):
        self.age += 1
        if self.age >= 5:
            self.alive = False
            
    def draw(self, surface, world_to_pixel_fn, turret_px):
        if not self.alive:
            return
        px, py = world_to_pixel_fn(self.x, self.y, turret_px)
        radius = int(2 + self.age * 2)
        color = (255, max(0, 255 - self.age * 40), 0)
        pygame.draw.circle(surface, color, (px, py), radius)
