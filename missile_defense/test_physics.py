import gymnasium as gym
import missile_defense
import numpy as np
import math

def test_physics():
    env = gym.make("MissileDefense-v0")
    env.reset()
    
    # Test 1: Parabolic arcs
    # Spawn a missile manually
    m = missile_defense.physics.Missile(x=-100, y=0, vx=20, vy=40)
    env.unwrapped.missiles.append(m)
    
    # Record positions
    positions = []
    for _ in range(50):
        env.step(np.array([0.0, -1.0])) # Do nothing
        positions.append((m.x, m.y))
        
    # Check if it follows parabola (Euler integration approximation)
    for i, (x, y) in enumerate(positions):
        n = i + 1 # step number
        dt = env.unwrapped.config.dt
        g = env.unwrapped.config.gravity
        expected_y = 0 + n * 40 * dt - g * (dt**2) * n * (n + 1) / 2
        assert abs(y - expected_y) < 1e-5, f"Expected {expected_y}, got {y}"
        
    print("Test 1 Passed: Parabolic arcs are correct.")
    
    # Test 2: Observation radar radius
    env.reset()
    m = missile_defense.physics.Missile(x=-100, y=100, vx=0, vy=0) # Distance 141 > 128
    env.unwrapped.missiles.append(m)
    obs, _, _, _, _ = env.step(np.array([0.0, -1.0]))
    
    # Check if missile is in obs
    # Since it's red (255, 50, 50), let's check for red pixels
    red_pixels = np.sum((obs[:, :, 0] == 255) & (obs[:, :, 1] == 50))
    assert red_pixels == 0, "Missile should not be visible"
    
    # Move it inside radar
    m.x = -50
    m.y = 50 # Distance 70 < 128
    obs, _, _, _, _ = env.step(np.array([0.0, -1.0]))
    red_pixels = np.sum((obs[:, :, 0] == 255) & (obs[:, :, 1] == 50))
    assert red_pixels > 0, "Missile should be visible"
    
    print("Test 2 Passed: Radar radius works.")
    
    # Test 3: Laser hit detection
    env.reset()
    m = missile_defense.physics.Missile(x=0, y=50, vx=0, vy=0) # Directly above turret
    env.unwrapped.missiles.append(m)
    
    # Turret angle is 90 (straight up)
    # Fire laser
    env.step(np.array([0.0, 1.0]))
    assert not m.alive, "Missile should be destroyed"
    
    # Test near miss
    env.reset()
    m = missile_defense.physics.Missile(x=5, y=50, vx=0, vy=0) # 5 units to the right
    env.unwrapped.missiles.append(m)
    
    # Hit radius is 2.5, so 5 units away should miss
    env.step(np.array([0.0, 1.0]))
    assert m.alive, "Missile should not be destroyed (near miss)"
    
    print("Test 3 Passed: Laser hit detection works.")
    
    # Test 4: Cooldown
    env.reset()
    env.step(np.array([0.0, 1.0])) # Fire
    assert env.unwrapped.last_laser_fired, "Laser should fire"
    
    env.step(np.array([0.0, 1.0])) # Fire again immediately
    assert not env.unwrapped.last_laser_fired, "Laser should be on cooldown"
    
    # Wait for cooldown
    for _ in range(env.unwrapped.config.cooldown - 1):
        env.step(np.array([0.0, -1.0]))
        
    env.step(np.array([0.0, 1.0])) # Fire again
    assert env.unwrapped.last_laser_fired, "Laser should fire after cooldown"
    
    print("Test 4 Passed: Cooldown works.")
    
    # Test 5: Difficulty ramp
    env.reset()
    initial_interval = env.unwrapped.current_spawn_interval
    
    # Manually increment kills to 1 less than ramp threshold
    env.unwrapped.kills = env.unwrapped.config.difficulty_ramp_every - 1
    
    # Trigger a hit to cause difficulty ramp
    m = missile_defense.physics.Missile(x=0, y=50, vx=0, vy=0)
    env.unwrapped.missiles.append(m)
    env.step(np.array([0.0, 1.0]))
    
    assert env.unwrapped.current_spawn_interval < initial_interval, "Difficulty should ramp up"
    
    print("Test 5 Passed: Difficulty ramps up.")
    
    # Test 6: Laser max range
    env.reset()
    m = missile_defense.physics.Missile(x=0, y=150, vx=0, vy=0) # Directly above turret, outside radar (128)
    env.unwrapped.missiles.append(m)
    env.step(np.array([0.0, 1.0])) # Fire
    assert m.alive, "Missile outside radar range should not be destroyed"
    
    print("Test 6 Passed: Laser max range works.")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_physics()
