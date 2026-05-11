import math
import numpy as np
import pygame
from .entity import Entity

class Turret(Entity):
    def __init__(self, config):
        self.config = config
        self.angle = 90.0  # Pointing straight up
        self.pos = (0.0, 0.0)
        self.last_laser_fired = False
        # Burst-fire mechanic state:
        #   - `burst_count`: number of consecutive frames the laser has fired so far
        #     in the current burst. Cleared to 0 whenever the laser does NOT fire
        #     (either voluntarily, because the agent didn't request fire, or due
        #     to a forced cooldown).
        #   - `forced_cooldown_remaining`: when this is > 0, the laser cannot fire
        #     this frame regardless of the agent's request. Decremented (and the
        #     burst counter cleared) at the start of any non-firing frame.
        self.burst_count = 0
        self.forced_cooldown_remaining = 0

    def reset(self):
        self.angle = 90.0
        self.last_laser_fired = False
        self.burst_count = 0
        self.forced_cooldown_remaining = 0

    def step(self, action_rot: float, action_fire: float) -> bool:
        """
        Updates turret state. Returns True if a laser is fired this step.
        action_rot: [-1, 1]
        action_fire: > 0 to fire (request only; may be blocked by forced cooldown)
        """
        # Update angle
        delta_angle = action_rot * self.config.omega_max
        self.angle += delta_angle
        self.angle = np.clip(self.angle, 0.0, 180.0)

        # Fire decision: allowed iff agent requested fire AND no forced cooldown
        # is currently in effect. If we fire, increment burst counter; if the
        # burst limit is reached on this frame, arm the forced cooldown so the
        # next frame is forced to skip firing.
        fired = False
        wants_fire = action_fire > 0
        if wants_fire and self.forced_cooldown_remaining == 0:
            fired = True
            self.burst_count += 1
            if self.burst_count >= self.config.burst_limit:
                self.forced_cooldown_remaining = self.config.burst_cooldown
                # `burst_count` will be reset on the next non-fire frame.
        else:
            # Either agent didn't request fire (voluntary stop) or the laser was
            # blocked by forced cooldown. Either way: burst resets, and we tick
            # down any pending forced cooldown.
            self.burst_count = 0
            if self.forced_cooldown_remaining > 0:
                self.forced_cooldown_remaining -= 1

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
