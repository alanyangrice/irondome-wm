import numpy as np
import pygame
import math

class Renderer:
    def __init__(self, config):
        self.config = config
        self.obs_w = config.obs_width
        self.obs_h = config.obs_height
        self.radar_r = config.radar_radius
        
        # Turret is at bottom-center of the observation
        self.turret_px = (self.obs_w // 2, self.obs_h)
        
        # Pygame setup
        pygame.init()
        self.obs_surface = pygame.Surface((self.obs_w, self.obs_h))
        self.human_surface = None
        self.clock = pygame.time.Clock()
        
        # Pre-compute mask for performance
        self.mask_idx = None
        
    def world_to_pixel(self, x: float, y: float, turret_px: tuple) -> tuple:
        """Converts world coordinates to pixel coordinates."""
        # 1 world unit = 1 pixel
        px = int(turret_px[0] + x)
        py = int(turret_px[1] - y)
        return (px, py)

    def render_obs(self, turret, missiles, laser_fired: bool, explosions: list) -> np.ndarray:
        """Renders the 256x128 observation image."""
        self.obs_surface.fill((0, 0, 0)) # Black background
        
        # Draw missiles and trails
        for m in missiles:
            if not m.alive:
                continue
            
            # Check if within radar
            dist_to_turret = math.hypot(m.x, m.y)
            if dist_to_turret > self.radar_r:
                continue
                
            # Draw trail
            if len(m.trail) > 1:
                pts = [self.world_to_pixel(tx, ty, self.turret_px) for tx, ty in m.trail]
                for i in range(len(pts) - 1):
                    alpha = (i + 1) / len(pts)
                    color = (int(200 * alpha), int(50 * alpha), int(100 * alpha))
                    pygame.draw.line(self.obs_surface, color, pts[i], pts[i+1], 1)
            
            # Draw missile
            px, py = self.world_to_pixel(m.x, m.y, self.turret_px)
            pygame.draw.circle(self.obs_surface, (255, 50, 50), (px, py), 2)
            
        # Draw turret
        # Base
        rect = pygame.Rect(self.turret_px[0] - 8, self.turret_px[1] - 8, 16, 8)
        pygame.draw.rect(self.obs_surface, (200, 200, 200), rect)
        
        # Barrel
        rad = math.radians(turret.angle)
        barrel_length = 12
        bx = int(self.turret_px[0] + math.cos(rad) * barrel_length)
        by = int(self.turret_px[1] - math.sin(rad) * barrel_length)
        pygame.draw.line(self.obs_surface, (150, 150, 150), self.turret_px, (bx, by), 3)
        
        # Draw laser
        if laser_fired:
            lx = int(self.turret_px[0] + math.cos(rad) * self.radar_r)
            ly = int(self.turret_px[1] - math.sin(rad) * self.radar_r)
            pygame.draw.line(self.obs_surface, (50, 255, 50), self.turret_px, (lx, ly), 1)
            
        # Draw explosions
        for ex, ey, age in explosions:
            px, py = self.world_to_pixel(ex, ey, self.turret_px)
            radius = int(2 + age * 2)
            color = (255, max(0, 255 - age * 40), 0)
            pygame.draw.circle(self.obs_surface, color, (px, py), radius)
            
        # Apply radar mask (semi-circle)
        img = pygame.surfarray.array3d(self.obs_surface)
        # pygame array is (W, H, C), we need (H, W, C)
        img = np.transpose(img, (1, 0, 2))
        
        if self.mask_idx is None:
            y, x = np.ogrid[:self.obs_h, :self.obs_w]
            dist_from_center = np.sqrt((x - self.turret_px[0])**2 + (y - self.turret_px[1])**2)
            self.mask_idx = dist_from_center > self.radar_r
            
        # Apply mask
        img[self.mask_idx] = 0
        
        return img

    def render_human(self, turret, missiles, laser_fired: bool, explosions: list, score: float, steps: int):
        """Renders a larger debug view showing the full world."""
        w, h = 400, 400
        
        if self.human_surface is None:
            self.human_surface = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Missile Defense")
            
        self.human_surface.fill((0, 0, 0))
        
        turret_px = (w // 2, h - 20)
        
        # Draw ground
        pygame.draw.line(self.human_surface, (100, 100, 100), (0, turret_px[1]), (w, turret_px[1]), 1)
        
        # Draw protected zone
        pz_min = self.world_to_pixel(-self.config.protected_zone_width/2, 0, turret_px)[0]
        pz_max = self.world_to_pixel(self.config.protected_zone_width/2, 0, turret_px)[0]
        pygame.draw.line(self.human_surface, (0, 255, 0), (pz_min, turret_px[1]), (pz_max, turret_px[1]), 3)
        
        # Draw radar boundary
        pygame.draw.circle(self.human_surface, (50, 50, 50), turret_px, int(self.radar_r), 1)
        
        # Draw missiles
        for m in missiles:
            if not m.alive:
                continue
            
            px, py = self.world_to_pixel(m.x, m.y, turret_px)
            
            # Trail
            if len(m.trail) > 1:
                pts = [self.world_to_pixel(tx, ty, turret_px) for tx, ty in m.trail]
                for i in range(len(pts) - 1):
                    pygame.draw.line(self.human_surface, (150, 50, 50), pts[i], pts[i+1], 1)
                    
            pygame.draw.circle(self.human_surface, (255, 0, 0), (px, py), 3)
            
            # Velocity vector
            vx_p, vy_p = self.world_to_pixel(m.x + m.vx*0.5, m.y + m.vy*0.5, turret_px)
            pygame.draw.line(self.human_surface, (200, 100, 100), (px, py), (vx_p, vy_p), 1)
            
        # Draw turret base
        rect = pygame.Rect(turret_px[0] - 10, turret_px[1] - 10, 20, 10)
        pygame.draw.rect(self.human_surface, (200, 200, 200), rect)
        
        # Draw turret barrel
        rad = math.radians(turret.angle)
        barrel_length = 15
        bx = int(turret_px[0] + math.cos(rad) * barrel_length)
        by = int(turret_px[1] - math.sin(rad) * barrel_length)
        pygame.draw.line(self.human_surface, (150, 150, 150), turret_px, (bx, by), 4)
        
        # Draw laser
        if laser_fired:
            lx = int(turret_px[0] + math.cos(rad) * self.radar_r)
            ly = int(turret_px[1] - math.sin(rad) * self.radar_r)
            pygame.draw.line(self.human_surface, (50, 255, 50), turret_px, (lx, ly), 1)
            
        # Draw explosions
        for ex, ey, age in explosions:
            px, py = self.world_to_pixel(ex, ey, turret_px)
            pygame.draw.circle(self.human_surface, (255, max(0, 255 - age*40), 0), (px, py), int(3 + age*3))
            
        # Text overlay
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {score:.1f}", True, (255, 255, 255))
        steps_text = font.render(f"Steps: {steps}", True, (255, 255, 255))
        cd_text = font.render(f"Cooldown: {turret.cooldown_timer}", True, (255, 255, 255))
        
        self.human_surface.blit(score_text, (10, 10))
        self.human_surface.blit(steps_text, (10, 30))
        self.human_surface.blit(cd_text, (10, 50))
        
        # Inset observation
        obs = self.render_obs(turret, missiles, laser_fired, explosions)
        # obs is (128, 256, 3) -> (H, W, C)
        # Convert back to pygame surface (W, H, C)
        obs_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Draw border
        pygame.draw.rect(self.human_surface, (255, 255, 255), (w-266, 10, 256, 128), 1)
        self.human_surface.blit(obs_surf, (w-266, 10))
        
        pygame.display.flip()
        self.clock.tick(self.config.render_fps)
        
        # Handle events to prevent window from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        return np.transpose(pygame.surfarray.array3d(self.human_surface), (1, 0, 2))

    def close(self):
        pygame.quit()
