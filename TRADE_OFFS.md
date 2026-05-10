## Trade-offs Made in the Environment

Here are the key design trade-offs and simplifications we made while building this environment, and why we made them:

### 1. Observation Space Size (512x256 vs 64x64)

- **Trade-off:** We chose a massive 512x256 observation space to perfectly fit a 256-unit radius radar semi-circle without wasting pixels or squishing the aspect ratio.
- **Why:** It makes the physics-to-pixel mapping exactly 1:1, making debugging and rendering much easier, and provides extremely sharp, high-resolution visuals for the agent.
- **Cost:** The original World Models paper used 64x64 images (4,096 pixels). Our images are 131,072 pixels (32x larger). This will require a massive VAE and significantly more compute to train the MDN-RNN. (Since we have an RTX 5090, compute is not a bottleneck).

### 2. Hitscan Laser vs Projectile Interceptors

- **Trade-off:** The turret fires an instantaneous "hitscan" laser beam rather than launching physical interceptor missiles (like a real Iron Dome).
- **Why:** It drastically simplifies the action space and collision detection. If the agent had to fire slow projectiles, it would need to calculate complex lead-targeting (shooting where the missile *will* be). Hitscan means the agent just has to point and click.
- **Cost:** Reduces realism. The agent doesn't have to learn to anticipate travel time for its own weapons.

### 3. Capped Terminal Velocity

- **Trade-off:** Missiles follow parabolic arcs, but their downward velocity (`vy`) is artificially capped at a `terminal_velocity` of 30.0.
- **Why:** Without a cap, missiles fired high into the air would accelerate under gravity for a long time and enter the 128-unit radar zone moving so fast that they cross the entire screen in just 2-3 frames. The agent (and the RNN) wouldn't have enough time to react or predict their path.
- **Cost:** The physics are slightly unnatural. Missiles stop accelerating downwards at a certain point, breaking pure parabolic motion near the end of long flights.

### 4. Variable Initial Launch Speeds

- **Trade-off:** All missiles are fired with the exact same initial speed (`v0 = 140.0`). Trajectory variation comes entirely from changing the launch angle.
- **Why:** It simplifies the spawning logic. We pick a random target in the radar zone, then calculate the two possible angles (high arc and low arc) that will hit that target at the fixed speed, and randomly pick one.
- **Cost:** Less diversity in the training data. The VAE and RNN won't have to learn to distinguish between "fast" and "slow" missile classes, only different angles.

### 5. Fixed Laser Cooldown

- **Trade-off:** The laser has a strict 5-step cooldown between shots, rather than an overheating mechanic or a continuous beam.
- **Why:** It prevents the agent from simply holding down the fire button and sweeping the turret back and forth to create an impenetrable wall of lasers. It forces the agent to be precise.
- **Cost:** The agent cannot rapidly correct a near-miss. If it fires and misses by 1 degree, it must wait 5 steps before trying again, by which time the missile might have hit the ground.

### 6. 2D Physics vs 3D Physics

- **Trade-off:** The entire simulation is strictly 2D (X and Y axes only).
- **Why:** 3D would require a 3D rendering engine (like OpenGL/MuJoCo) and would make the observation space much more complex to encode. 2D allows us to use simple Pygame drawing primitives and keeps the environment blazingly fast (>1400 FPS).
- **Cost:** Removes the concept of depth. Missiles cannot fly "over" or "behind" the turret, they can only fly left/right/up/down.

### 7. Radar Range vs Laser Range

- **Trade-off:** The radar can see up to 256 units, but the laser can only shoot up to 128 units.
- **Why:** This creates a "tracking zone". The agent can see the missile entering the radar, but must wait for it to get closer before firing. This forces the RNN to track the missile's trajectory in memory while waiting for it to enter the kill zone, rather than just reflexively shooting it the moment it appears on screen.
- **Cost:** Makes the task harder for the agent, as it requires patience and timing rather than just pointing and clicking immediately.

