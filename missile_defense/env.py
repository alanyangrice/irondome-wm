import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from missile_defense.config import DEFAULT_CONFIG
from missile_defense.physics import check_laser_hits
from missile_defense.entities import Turret, spawn_missile, spawn_cloud, spawn_bird, Explosion, City
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
        self.city = City(self.config)
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
        reward = self.config.step_penalty
        terminated = False
        truncated = False
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        action_rot = float(action[0])
        action_fire = float(action[1])
        
        # 1. Update Turret
        self.last_laser_fired = self.turret.step(action_rot, action_fire)
        if self.last_laser_fired:
            reward += self.config.fire_penalty
            
        # 2. Spawn Missiles
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and len(self.missiles) < self.config.max_missiles:
            self.missiles.append(spawn_missile(self.config, self.np_random))
            self.spawn_timer = self.current_spawn_interval
            
        # 3. Update Missiles
        for m in self.missiles:
            m.step(self.config)
            
        # 4. Check Laser Hits
        # iter3 semantics: each hit pays `hit_reward`. If the hit reduces the
        # missile's HP to zero, it explodes (set alive=False, spawn explosion,
        # award `kill_reward` bonus on top of `hit_reward`, count kill, possibly
        # ramp difficulty). Otherwise the missile keeps falling at reduced HP.
        # `check_laser_hits` no longer mutates the missile; we do all updates
        # here so reward bookkeeping is in one place.
        if self.last_laser_fired:
            hit_idx = check_laser_hits(self.turret, self.missiles, self.config)
            if hit_idx != -1:
                m = self.missiles[hit_idx]
                m.hp -= 1
                reward += self.config.hit_reward
                if m.hp <= 0:
                    reward += self.config.kill_reward
                    m.alive = False
                    self.kills += 1
                    self.explosions.append(Explosion(m.x, m.y))

                    # Difficulty ramp on full kills only (not on intermediate hits).
                    if self.kills % self.config.difficulty_ramp_every == 0:
                        self.current_spawn_interval = max(
                            self.config.min_spawn_interval,
                            int(self.current_spawn_interval * 0.9)
                        )
                    
        # 5. Check Ground Impacts (and drop laser-killed missiles)
        alive_missiles = []
        for m in self.missiles:
            if not m.alive:
                # Killed by laser earlier this step; drop from list so max_missiles isn't permanently saturated.
                continue
            if m.y <= 0:
                m.alive = False
                # Check if it hit the protected zone
                if abs(m.x) <= self.config.protected_zone_width / 2:
                    reward += self.config.protected_zone_penalty
                    terminated = True
                else:
                    reward += self.config.non_protected_impact_reward
                
                # Add explosion on ground
                self.explosions.append(Explosion(m.x, 0))
            else:
                alive_missiles.append(m)
                
        self.missiles = alive_missiles
        
        # 6. Update Explosions
        for ex in self.explosions:
            ex.step(self.config)
        self.explosions = [ex for ex in self.explosions if ex.alive]
        
        # 7. Update Clouds and Birds
        for c in self.clouds:
            c.step(self.config)
        self.clouds = [c for c in self.clouds if abs(c.x) < self.config.radar_radius + 150]
            
        while len(self.clouds) < self.config.max_clouds:
            self.clouds.append(spawn_cloud(self.config, self.np_random))
            
        for b in self.birds:
            b.step(self.config)
        self.birds = [b for b in self.birds if abs(b.x) < self.config.radar_radius + 150]
            
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
        return self.renderer.render_obs(self.turret, self.missiles, self.last_laser_fired, self.explosions, self.clouds, self.birds, self.city)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            img = self.renderer.render_human(
                self.turret, self.missiles, self.last_laser_fired, self.explosions, self.clouds, self.birds, self.city, self.score, self.steps
            )
            return img

    def close(self):
        self.renderer.close()
