import math
import numpy as np
import pygame
from .entity import Entity

class Turret(Entity):
    def __init__(self, config):
        self.config = config
        self.angle = 90.0  # Pointing straight up
        self.cooldown_timer = 0
        self.pos = (0.0, 0.0)
        self.last_laser_fired = False

    def reset(self):
        self.angle = 90.0
        self.cooldown_timer = 0
        self.last_laser_fired = False

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

        self.last_laser_fired = fired
        return fired
        
    def draw(self, surface, world_to_pixel_fn, turret_px):
        # Base
        rect = pygame.Rect(turret_px[0] - 10, turret_px[1] - 10, 20, 10)
        pygame.draw.rect(surface, (200, 200, 200), rect)

        # Barrel pivot point (top center of base)
        pivot = (turret_px[0], turret_px[1] - 10)
        rad = math.radians(self.angle)
        barrel_length = 15
        bx = int(pivot[0] + math.cos(rad) * barrel_length)
        by = int(pivot[1] - math.sin(rad) * barrel_length)
        pygame.draw.line(surface, (150, 150, 150), pivot, (bx, by), 4)
        
        # Draw laser
        if self.last_laser_fired:
            lx = int(pivot[0] + math.cos(rad) * self.config.laser_radius)
            ly = int(pivot[1] - math.sin(rad) * self.config.laser_radius)
            pygame.draw.line(surface, (50, 255, 50), pivot, (lx, ly), 1)
