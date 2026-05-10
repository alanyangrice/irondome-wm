# Missile Defense Environment - TODOs & Limitations

## Future Improvements

### Graphics & Visuals (For VAE Encoding)
- **Detailed Missile Representations:** Replace simple dots with directional vector shapes (e.g., triangles or rocket shapes) that rotate to match their velocity vector.
- **Turret Details:** Upgrade the turret from a simple geometric primitive to a detailed multi-part polygon (base, rotating barrel, radar dish).
- **Particle Effects:** Add more sophisticated particle systems for explosions and laser impacts.
- **Background Details:** Add ground textures or a faint grid to help the VAE encode spatial relationships better.
- **Sprite Support:** Consider allowing the environment to load pixel art sprites for entities instead of drawing vector shapes.

### Physics & Gameplay
- **Variable Missile Speeds:** Introduce different classes of missiles with varying initial speeds and terminal velocities.
- **Multiple Turrets:** Support for multi-turret setups to defend larger areas.
- **Wind/Drag:** Add atmospheric drag or wind to make trajectories less perfectly parabolic and harder to predict.

## Current Limitations
- **Simple Visuals:** Currently using basic geometric shapes (dots, lines, simple polygons) which might limit the complexity of features the VAE can learn.
- **Fixed Initial Speed:** All missiles are fired with the same initial speed `v0`, relying only on angle variations for trajectory diversity.
- **Hitscan Laser:** The laser is instantaneous (hitscan) rather than a projectile, simplifying interception logic but reducing realism.
- **Laser Cooldown:** The laser has a fixed cooldown (currently 5 steps). We might want to revisit this later to allow for continuous beam firing or an overheating mechanic instead of a strict step-based cooldown.
- **2D Only:** The simulation is strictly 2D, ignoring depth.
- **Capped Terminal Velocity:** Physics are slightly unnatural due to capping the falling velocity, which is necessary to ensure the agent has enough reaction time within the 128-radius radar.
