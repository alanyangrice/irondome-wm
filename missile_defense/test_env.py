import gymnasium as gym
import missile_defense
import time

def test_env():
    env = gym.make("MissileDefense-v0", render_mode="human")
    obs, info = env.reset()
    
    # Run a few episodes
    for ep in range(3):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Bias action to fire occasionally
            if env.np_random.random() < 0.9:
                action[1] = -1.0 # Don't fire
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            
            # Control frame rate for human viewing is handled by renderer.clock.tick in render()
            # But we need to call render() if it's not called in step (it is called in step if render_mode is human)
            
        print(f"Episode {ep} finished. Score: {info['score']:.1f}, Kills: {info['kills']}")
        
    env.close()

if __name__ == "__main__":
    test_env()
