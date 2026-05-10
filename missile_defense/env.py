import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from missile_defense.config import DEFAULT_CONFIG
from missile_defense.physics import Turret, spawn_missile, check_laser_hits, spawn_cloud, spawn_bird
from missile_defense.renderer import Renderer

class MissileDefenseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", config=None):
        super().__init__()
        self.config = config if config is not None else DEFAULT_CONFIG
        self.render_mode = render_mode
        
        # Observation space: 128x256 RGB image (H, W, C)
        self.observation_space = Box(
            low=0, high=255, 
            shape=(self.config.obs_height, self.config.obs_width, 3), 
            dtype=np.uint8
        )
        
        # Action space: [rotation, fire]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.renderer = Renderer(self.config)
        
        self.turret = None
        self.missiles = []
        self.explosions = [] # list of [x, y, age]
        self.clouds = []
        self.birds = []
        
        self.steps = 0
        self.score = 0.0
        self.spawn_timer = 0
        self.current_spawn_interval = self.config.spawn_interval
        self.kills = 0

        self.last_laser_fired = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.turret = Turret(self.config)
        self.missiles = []
        self.explosions = []
        self.clouds = []
        self.birds = []
        
        # Pre-populate some clouds and birds
        for _ in range(self.config.max_clouds):
            self.clouds.append(spawn_cloud(self.config, self.np_random))
            # Randomize their initial x positions across the screen
            self.clouds[-1].x = self.np_random.uniform(-self.config.radar_radius, self.config.radar_radius)
            
        for _ in range(self.config.max_birds):
            self.birds.append(spawn_bird(self.config, self.np_random))
            self.birds[-1].x = self.np_random.uniform(-self.config.radar_radius, self.config.radar_radius)
        
        self.steps = 0
        self.score = 0.0
        self.kills = 0
        self.current_spawn_interval = self.config.spawn_interval
        self.spawn_timer = 10 # Initial delay
        
        self.last_laser_fired = False
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        self.steps += 1
        reward = -0.01 # Time penalty
        terminated = False
        truncated = False
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        action_rot = float(action[0])
        action_fire = float(action[1])
        
        # 1. Update Turret
        self.last_laser_fired = self.turret.step(action_rot, action_fire)
        if self.last_laser_fired:
            reward -= 0.05 # Conservation penalty
            
        # 2. Spawn Missiles
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and len(self.missiles) < self.config.max_missiles:
            self.missiles.append(spawn_missile(self.config, self.np_random))
            self.spawn_timer = self.current_spawn_interval
            
        # 3. Update Missiles
        for m in self.missiles:
            m.step(self.config)
            
        # 4. Check Laser Hits
        if self.last_laser_fired:
            hit_idx = check_laser_hits(self.turret, self.missiles, self.config)
            if hit_idx != -1:
                reward += 1.0
                self.kills += 1
                
                # Add explosion
                m = self.missiles[hit_idx]
                self.explosions.append([m.x, m.y, 0])
                
                # Difficulty ramp
                if self.kills % self.config.difficulty_ramp_every == 0:
                    self.current_spawn_interval = max(
                        self.config.min_spawn_interval, 
                        int(self.current_spawn_interval * 0.9)
                    )
                    
        # 5. Check Ground Impacts
        alive_missiles = []
        for m in self.missiles:
            if not m.alive:
                continue
                
            if m.y <= 0:
                # Check if it hit the protected zone
                if abs(m.x) <= self.config.protected_zone_width / 2:
                    reward -= 10.0
                    terminated = True
                else:
                    # Ignored non-threat
                    pass
                
                # Add explosion on ground
                self.explosions.append([m.x, 0, 0])
            else:
                alive_missiles.append(m)
                
        self.missiles = alive_missiles
        
        # 6. Update Explosions
        new_explosions = []
        for ex in self.explosions:
            ex[2] += 1 # age
            if ex[2] < 5:
                new_explosions.append(ex)
        self.explosions = new_explosions
        
        # 7. Update Clouds and Birds
        alive_clouds = []
        for c in self.clouds:
            c.step(self.config)
            if abs(c.x) < self.config.radar_radius + 150:
                alive_clouds.append(c)
        self.clouds = alive_clouds
        
        while len(self.clouds) < self.config.max_clouds:
            self.clouds.append(spawn_cloud(self.config, self.np_random))
            
        alive_birds = []
        for b in self.birds:
            b.step(self.config)
            if abs(b.x) < self.config.radar_radius + 150:
                alive_birds.append(b)
        self.birds = alive_birds
        
        while len(self.birds) < self.config.max_birds:
            self.birds.append(spawn_bird(self.config, self.np_random))
        
        # 8. Check Truncation
        if self.steps >= self.config.max_steps:
            truncated = True
            
        self.score += reward
        
        obs = self._get_obs()
        info = {"score": self.score, "kills": self.kills}
        
        # Render if human mode
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return self.renderer.render_obs(self.turret, self.missiles, self.last_laser_fired, self.explosions, self.clouds, self.birds)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            img = self.renderer.render_human(
                self.turret, self.missiles, self.last_laser_fired, self.explosions, self.clouds, self.birds, self.score, self.steps
            )
            return img

    def close(self):
        self.renderer.close()
