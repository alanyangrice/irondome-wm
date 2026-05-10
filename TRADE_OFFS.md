## Trade-offs Made in the Environment

Here are the key design trade-offs and simplifications we made while building this environment, and why we made them:

### 1. Observation Space Size (512×256 stored; VAE sees less)

- **Simulator:** Observations are **512×256** RGB so the **256-unit** radar maps cleanly to pixels, debugging stays sharp, and collected `.npz` rollouts stay at native resolution.
- **World model:** The PyTorch dataloader **downsamples bilinearly** (default **128×64**, same aspect ratio as **512×256**, configurable) before the VAE/MDN-RNN, matching **Ha & Schmidhuber (2018)**-style **compact** inputs without re-rendering or re-collecting data.
- **Cost (simulator):** More disk and framebuffer than the paper’s **64×64** CarRacing crop (paper: 4,096 pixels per frame vs our 131,072—about **32×**).
- **Cost (vision model):** Kept small by training on resized tensors, not on full 512×256.

### 2. Hitscan Laser vs Projectile Interceptors

- **Trade-off:** The turret fires an instantaneous "hitscan" laser beam rather than launching physical interceptor missiles (like a real Iron Dome).
- **Why:** It drastically simplifies the action space and collision detection. If the agent had to fire slow projectiles, it would need to calculate complex lead-targeting (shooting where the missile *will* be). Hitscan means the agent just has to point and click.
- **Cost:** Reduces realism. The agent doesn't have to learn to anticipate travel time for its own weapons.

### 3. Capped Terminal Velocity

- **Trade-off:** Missiles follow parabolic arcs, but their downward velocity (`vy`) is **capped** at **`terminal_velocity`** (currently **40.0** in `missile_defense/config.py`).
- **Why:** Without a cap, missiles fired high into the air would accelerate under gravity for a long time and enter the **256-unit** radar zone moving so fast that they cross the screen in very few steps. The agent (and the RNN) would have too little time to react or predict their path.
- **Cost:** The physics are slightly unnatural. Missiles stop accelerating downwards past the cap, breaking pure parabolic motion on long drop segments.

### 4. Fixed Initial Launch Speed

- **Trade-off:** All missiles are fired with the same initial speed **`missile_v0`** (currently **65.0**). Trajectory variation comes mainly from launch angle (clamped by **`min_spawn_angle`** / **`max_spawn_angle`**).
- **Why:** Simpler spawning and more stable difficulty tuning.
- **Cost:** Less diversity in speed classes; the model may face an easier distribution of closing speeds.

### 5. Fixed Laser Cooldown

- **Trade-off:** The laser has a **strict step cooldown** (**`cooldown`**, currently **2** steps in `config.py`) between shots.
- **Why:** Prevents trivial “spray” policies that sweep a continuous beam and makes timing matter.
- **Cost:** The agent cannot rapidly correct a near-miss; it must wait for the next firing window.

### 6. 2D Physics vs 3D Physics

- **Trade-off:** The entire simulation is strictly 2D (X and Y axes only).
- **Why:** 3D would require a heavier stack and a richer observation. 2D keeps rollout throughput high.
- **Cost:** No depth or “behind” geometry—only planar motion.

### 7. Radar Range vs Laser Range

- **Trade-off:** The radar sees out to **`radar_radius`** (**256** units) while the laser only reaches **`laser_radius`** (**64** units)—an **8:1** ratio.
- **Why:** Creates a **tracking zone**: the agent can see threats before they are engageable, which encourages memory and timing (aligned with partial observability / world-model motivation).
- **Cost:** Harder than equal-range play: firing too early wastes shots; the agent must track into the kill zone.
