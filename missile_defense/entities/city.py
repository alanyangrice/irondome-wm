import pygame
from .entity import Entity

class City(Entity):
    def __init__(self, config):
        self.config = config
        
    def step(self, config):
        pass # City is static
        
    def draw(self, surface, world_to_pixel_fn, turret_px):
        pz_min = world_to_pixel_fn(-self.config.protected_zone_width/2, 0, turret_px)[0]
        pz_max = world_to_pixel_fn(self.config.protected_zone_width/2, 0, turret_px)[0]
        
        # Draw a few simple buildings in the protected zone
        building_colors = [(100, 100, 110), (80, 80, 90), (120, 120, 130)]
        building_widths = [6, 8, 5, 7, 9]
        building_heights = [10, 18, 8, 20, 12]
        
        current_x = pz_min + 1
        for i in range(10): # Increased loop count to cover full width
            if current_x + building_widths[i % len(building_widths)] > pz_max:
                break
            
            bw = building_widths[i % len(building_widths)]
            bh = building_heights[i % len(building_heights)]
            bc = building_colors[i % len(building_colors)]
            
            # Draw building rectangle
            brect = pygame.Rect(current_x, turret_px[1] - bh, bw, bh)
            pygame.draw.rect(surface, bc, brect)
            
            # Draw some windows
            for wy in range(turret_px[1] - bh + 2, turret_px[1] - 2, 4):
                for wx in range(current_x + 1, current_x + bw - 1, 3):
                    if (wx * wy) % 3 == 0:
                        pygame.draw.rect(surface, (200, 200, 100), (wx, wy, 1, 2))
                    else:
                        pygame.draw.rect(surface, (40, 40, 50), (wx, wy, 1, 2))
                        
            current_x += bw + 1
            
        pygame.draw.line(surface, (0, 200, 0), (pz_min, turret_px[1]), (pz_max, turret_px[1]), 4)
