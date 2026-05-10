class Entity:
    """Base class for all entities in the environment."""
    
    def step(self, config):
        """Update the entity's state."""
        pass
        
    def draw(self, surface, world_to_pixel_fn, turret_px):
        """Render the entity to the given surface."""
        pass
