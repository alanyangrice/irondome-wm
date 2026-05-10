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

    def render_obs(self, turret, missiles, laser_fired: bool, explosions: list, clouds: list, birds: list) -> np.ndarray:
        """Renders the 512x256 observation image."""
        self.obs_surface.fill((15, 20, 30)) # Very dark blue background instead of pure black
        
        # Draw faint background grid
        grid_spacing = 32
        for x in range(0, self.obs_w, grid_spacing):
            pygame.draw.line(self.obs_surface, (30, 30, 40), (x, 0), (x, self.obs_h), 1)
        for y in range(0, self.obs_h, grid_spacing):
            pygame.draw.line(self.obs_surface, (30, 30, 40), (0, y), (self.obs_w, y), 1)
            
        # Draw clouds
        for c in clouds:
            px, py = self.world_to_pixel(c.x, c.y, self.turret_px)
            # Draw a fluffy cloud using multiple circles
            pygame.draw.circle(self.obs_surface, (60, 60, 70), (px, py), int(c.size))
            pygame.draw.circle(self.obs_surface, (60, 60, 70), (px + int(c.size*0.6), py - int(c.size*0.3)), int(c.size*0.8))
            pygame.draw.circle(self.obs_surface, (60, 60, 70), (px - int(c.size*0.6), py - int(c.size*0.3)), int(c.size*0.7))
            pygame.draw.circle(self.obs_surface, (60, 60, 70), (px + int(c.size*1.1), py), int(c.size*0.6))
            pygame.draw.circle(self.obs_surface, (60, 60, 70), (px - int(c.size*1.1), py), int(c.size*0.5))
            
        # Draw birds
        for b in birds:
            px, py = self.world_to_pixel(b.x, b.y, self.turret_px)
            # Flapping wings based on b.t
            wing_offset = int(math.sin(b.t * b.vy_freq * 2) * 5)
            pygame.draw.line(self.obs_surface, (120, 120, 120), (px, py), (px - 6, py - 3 + wing_offset), 2)
            pygame.draw.line(self.obs_surface, (120, 120, 120), (px, py), (px + 6, py - 3 + wing_offset), 2)
        
        # Draw missiles and trails
        for m in missiles:
            if not m.alive:
                continue
            
            # Check if within radar
            dist_to_turret = math.hypot(m.x, m.y)
            if dist_to_turret > self.radar_r:
                continue
                
            # Check if blocked by a cloud
            blocked = False
            for c in clouds:
                dist_to_cloud = math.hypot(m.x - c.x, m.y - c.y)
                if dist_to_cloud < c.size * 1.2: # Approximate cloud radius
                    blocked = True
                    break
                    
            if blocked:
                continue
                
            # Draw trail
            if len(m.trail) > 1:
                pts = [self.world_to_pixel(tx, ty, self.turret_px) for tx, ty in m.trail]
                for i in range(len(pts) - 1):
                    alpha = (i + 1) / len(pts)
                    color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))
                    pygame.draw.line(self.obs_surface, color, pts[i], pts[i+1], 3) # Thicker trail
            
            # Draw missile
            px, py = self.world_to_pixel(m.x, m.y, self.turret_px)
            pygame.draw.circle(self.obs_surface, (255, 100, 50), (px, py), 4) # Thicker missile
            # Inner core
            pygame.draw.circle(self.obs_surface, (255, 255, 200), (px, py), 2)
            
        # Draw ground
        ground_rect = pygame.Rect(0, self.turret_px[1], self.obs_w, self.obs_h - self.turret_px[1])
        pygame.draw.rect(self.obs_surface, (20, 30, 20), ground_rect) # Dark green ground
        pygame.draw.line(self.obs_surface, (50, 80, 50), (0, self.turret_px[1]), (self.obs_w, self.turret_px[1]), 2)
        
        # Draw protected zone
        pz_min = self.world_to_pixel(-self.config.protected_zone_width/2, 0, self.turret_px)[0]
        pz_max = self.world_to_pixel(self.config.protected_zone_width/2, 0, self.turret_px)[0]
        pygame.draw.line(self.obs_surface, (0, 200, 0), (pz_min, self.turret_px[1]), (pz_max, self.turret_px[1]), 4)
        
        # Base
        rect = pygame.Rect(self.turret_px[0] - 8, self.turret_px[1] - 8, 16, 8)
        pygame.draw.rect(self.obs_surface, (200, 200, 200), rect)

        # Barrel pivot point (top center of base)
        pivot = (self.turret_px[0], self.turret_px[1] - 8)
        rad = math.radians(turret.angle)
        barrel_length = 12
        bx = int(pivot[0] + math.cos(rad) * barrel_length)
        by = int(pivot[1] - math.sin(rad) * barrel_length)
        pygame.draw.line(self.obs_surface, (150, 150, 150), pivot, (bx, by), 3)
        
        # Draw laser
        if laser_fired:
            lx = int(pivot[0] + math.cos(rad) * self.config.laser_radius)
            ly = int(pivot[1] - math.sin(rad) * self.config.laser_radius)
            pygame.draw.line(self.obs_surface, (50, 255, 50), pivot, (lx, ly), 1)
            
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

    def render_human(self, turret, missiles, laser_fired: bool, explosions: list, clouds: list, birds: list, score: float, steps: int):
        """Renders a larger debug view showing the full world."""
        w, h = 800, 800
        
        if self.human_surface is None:
            self.human_surface = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Missile Defense")
            
        self.human_surface.fill((15, 20, 30))
        
        turret_px = (w // 2, h - 20)
        
        # Draw background grid
        grid_spacing = 64
        for x in range(0, w, grid_spacing):
            pygame.draw.line(self.human_surface, (30, 30, 40), (x, 0), (x, h), 1)
        for y in range(0, h, grid_spacing):
            pygame.draw.line(self.human_surface, (30, 30, 40), (0, y), (w, y), 1)
            
        # Draw clouds
        for c in clouds:
            px, py = self.world_to_pixel(c.x, c.y, turret_px)
            pygame.draw.circle(self.human_surface, (60, 60, 70), (px, py), int(c.size))
            pygame.draw.circle(self.human_surface, (60, 60, 70), (px + int(c.size*0.6), py - int(c.size*0.3)), int(c.size*0.8))
            pygame.draw.circle(self.human_surface, (60, 60, 70), (px - int(c.size*0.6), py - int(c.size*0.3)), int(c.size*0.7))
            pygame.draw.circle(self.human_surface, (60, 60, 70), (px + int(c.size*1.1), py), int(c.size*0.6))
            pygame.draw.circle(self.human_surface, (60, 60, 70), (px - int(c.size*1.1), py), int(c.size*0.5))
            
        # Draw birds
        for b in birds:
            px, py = self.world_to_pixel(b.x, b.y, turret_px)
            wing_offset = int(math.sin(b.t * b.vy_freq * 2) * 5)
            pygame.draw.line(self.human_surface, (120, 120, 120), (px, py), (px - 6, py - 3 + wing_offset), 2)
            pygame.draw.line(self.human_surface, (120, 120, 120), (px, py), (px + 6, py - 3 + wing_offset), 2)
        
        # Draw ground
        ground_rect = pygame.Rect(0, turret_px[1], w, h - turret_px[1])
        pygame.draw.rect(self.human_surface, (20, 30, 20), ground_rect)
        pygame.draw.line(self.human_surface, (50, 80, 50), (0, turret_px[1]), (w, turret_px[1]), 2)
        
        # Draw protected zone
        pz_min = self.world_to_pixel(-self.config.protected_zone_width/2, 0, turret_px)[0]
        pz_max = self.world_to_pixel(self.config.protected_zone_width/2, 0, turret_px)[0]
        pygame.draw.line(self.human_surface, (0, 200, 0), (pz_min, turret_px[1]), (pz_max, turret_px[1]), 4)
        
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

        # Barrel pivot point (top center of base)
        pivot = (turret_px[0], turret_px[1] - 10)
        rad = math.radians(turret.angle)
        barrel_length = 15
        bx = int(pivot[0] + math.cos(rad) * barrel_length)
        by = int(pivot[1] - math.sin(rad) * barrel_length)
        pygame.draw.line(self.human_surface, (150, 150, 150), pivot, (bx, by), 4)
        
        # Draw laser
        if laser_fired:
            lx = int(pivot[0] + math.cos(rad) * self.config.laser_radius)
            ly = int(pivot[1] - math.sin(rad) * self.config.laser_radius)
            pygame.draw.line(self.human_surface, (50, 255, 50), pivot, (lx, ly), 1)
            
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
        obs = self.render_obs(turret, missiles, laser_fired, explosions, clouds, birds)
        # obs is (128, 256, 3) -> (H, W, C)
        # Convert back to pygame surface (W, H, C)
        obs_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Draw border
        pygame.draw.rect(self.human_surface, (255, 255, 255), (w-522, 10, 512, 256), 1)
        self.human_surface.blit(obs_surf, (w-522, 10))
        
        pygame.display.flip()
        self.clock.tick(self.config.render_fps)
        
        # Handle events to prevent window from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        return np.transpose(pygame.surfarray.array3d(self.human_surface), (1, 0, 2))

    def close(self):
        pygame.quit()
