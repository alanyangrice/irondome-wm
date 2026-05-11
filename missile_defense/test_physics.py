import gymnasium as gym
import missile_defense
from missile_defense.entities import Missile
import numpy as np
import math

def test_physics():
    env = gym.make("MissileDefense-v0")
    env.reset()
    
    # Test 1: Parabolic arcs
    # Spawn a missile manually
    m = Missile(x=-100, y=0, vx=20, vy=40)
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
        assert abs(y - expected_y) < 1e-4, f"Expected {expected_y}, got {y}"
        
    print("Test 1 Passed: Parabolic arcs are correct.")
    
    # Test 2: Observation radar radius
    env.reset()
    m = Missile(x=-100, y=300, vx=0, vy=0) # Distance 316 > 256
    env.unwrapped.missiles.append(m)
    obs, _, _, _, _ = env.step(np.array([0.0, -1.0]))
    
    # Check if missile is in obs
    # Since it's red (255, 50, 50), let's check for red pixels
    red_pixels = np.sum((obs[:, :, 0] == 255) & (obs[:, :, 1] == 50))
    assert red_pixels == 0, "Missile should not be visible"
    
    # Move it inside radar
    m.x = -50
    m.y = 50 # Distance 70 < 256. Also y=50 is below cloud_min_y (100) so it won't be blocked by random clouds
    # Clear clouds for this test to ensure it's not randomly blocked
    env.unwrapped.clouds = []
    obs, _, _, _, _ = env.step(np.array([0.0, -1.0]))
    red_pixels = np.sum((obs[:, :, 0] == 255) & (obs[:, :, 1] == 100)) # Missile core is (255, 100, 50)
    assert red_pixels > 0, "Missile should be visible"
    
    print("Test 2 Passed: Radar radius works.")
    
    # Test 3: Laser hit detection (HP system, iter3)
    env.reset()
    max_hp = env.unwrapped.config.missile_hp
    m = Missile(x=0, y=20, vx=0, vy=0, max_hp=max_hp)  # Directly above turret, in range
    env.unwrapped.missiles.append(m)

    # First hit: should damage but NOT kill.
    env.step(np.array([0.0, 1.0]))
    assert m.alive, "Missile should still be alive after one hit (HP system)"
    assert m.hp == max_hp - 1, f"Missile HP should be {max_hp - 1} after one hit, got {m.hp}"

    # Remaining hits: should kill on the final one. Burst limit is 5, HP is 3,
    # so all hits land in one burst with no forced cooldown.
    for _ in range(max_hp - 1):
        env.step(np.array([0.0, 1.0]))
    assert not m.alive, f"Missile should be dead after {max_hp} hits"

    # Test near miss (still uses one shot since no HP is decremented on a miss).
    env.reset()
    m = Missile(x=5, y=20, vx=0, vy=0, max_hp=max_hp)  # 5 units to the right
    env.unwrapped.missiles.append(m)
    env.step(np.array([0.0, 1.0]))  # Hit radius 2.5, so 5 units away should miss
    assert m.alive, "Missile should not be damaged (near miss)"
    assert m.hp == max_hp, "Missile HP should be unchanged on a near miss"

    print("Test 3 Passed: Laser hit detection + HP system works.")

    # Test 4: Burst-fire mechanic (iter3 replacement for Test 4: Cooldown)
    env.reset()
    limit = env.unwrapped.config.burst_limit
    cd = env.unwrapped.config.burst_cooldown

    # Fire `limit` shots in a row: all should fire.
    for i in range(limit):
        env.step(np.array([0.0, 1.0]))
        assert env.unwrapped.last_laser_fired, f"Shot {i+1}/{limit} should fire"

    # Next `cd` frames: agent commands fire, but laser is in forced cooldown.
    for i in range(cd):
        env.step(np.array([0.0, 1.0]))
        assert not env.unwrapped.last_laser_fired, f"Forced cooldown frame {i+1}/{cd} should NOT fire"

    # After forced cooldown, a full burst is available again.
    for i in range(limit):
        env.step(np.array([0.0, 1.0]))
        assert env.unwrapped.last_laser_fired, f"Post-cooldown shot {i+1}/{limit} should fire"

    # Voluntary stop resets the counter: fire 3, skip 1 voluntarily, then fire `limit` more.
    env.reset()
    for _ in range(3):
        env.step(np.array([0.0, 1.0]))
        assert env.unwrapped.last_laser_fired
    env.step(np.array([0.0, -1.0]))  # voluntary skip
    assert not env.unwrapped.last_laser_fired
    for i in range(limit):
        env.step(np.array([0.0, 1.0]))
        assert env.unwrapped.last_laser_fired, (
            f"After voluntary stop, burst should be full again; shot {i+1}/{limit} failed"
        )

    print("Test 4 Passed: Burst-fire mechanic works.")

    # Test 5: Difficulty ramp (HP system: full kill triggers ramp, not first hit)
    env.reset()
    initial_interval = env.unwrapped.current_spawn_interval

    # Manually increment kills to 1 less than ramp threshold
    env.unwrapped.kills = env.unwrapped.config.difficulty_ramp_every - 1

    # Trigger a full kill: requires `missile_hp` consecutive hits
    m = Missile(x=0, y=20, vx=0, vy=0, max_hp=env.unwrapped.config.missile_hp)
    env.unwrapped.missiles.append(m)
    for _ in range(env.unwrapped.config.missile_hp):
        env.step(np.array([0.0, 1.0]))
    assert not m.alive, "Missile should be killed for ramp test"

    assert env.unwrapped.current_spawn_interval < initial_interval, "Difficulty should ramp up"

    print("Test 5 Passed: Difficulty ramps up.")
    
    # Test 6: Laser max range
    env.reset()
    m = Missile(x=0, y=80, vx=0, vy=0) # Directly above turret, outside laser (32) but inside radar (256)
    # Clear clouds so it doesn't get blocked
    env.unwrapped.clouds = []
    env.unwrapped.missiles.append(m)
    env.step(np.array([0.0, 1.0])) # Fire
    assert m.alive, "Missile outside laser range should not be destroyed"
    
    print("Test 6 Passed: Laser max range works.")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_physics()
