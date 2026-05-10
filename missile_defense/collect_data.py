import gymnasium as gym
import missile_defense
import numpy as np
import os
import argparse
import time

def collect_data(num_episodes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    env = gym.make("MissileDefense-v0", render_mode="rgb_array")
    
    total_steps = 0
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        observations = []
        actions = []
        rewards = []
        dones = []
        
        done = False
        while not done:
            action = env.action_space.sample()
            
            # Bias action to fire occasionally
            if env.np_random.random() < 0.9:
                action[1] = -1.0 # Don't fire
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            done = terminated or truncated
            dones.append(done)
            
            obs = next_obs
            total_steps += 1
            
        # Save episode data
        np.savez_compressed(
            os.path.join(output_dir, f"ep_{ep}.npz"),
            obs=np.array(observations, dtype=np.uint8),
            action=np.array(actions, dtype=np.float32),
            reward=np.array(rewards, dtype=np.float32),
            done=np.array(dones, dtype=bool)
        )
        
        if (ep + 1) % 10 == 0:
            print(f"Collected {ep + 1}/{num_episodes} episodes. Total steps: {total_steps}")
            
    end_time = time.time()
    fps = total_steps / (end_time - start_time)
    print(f"Data collection complete. Total steps: {total_steps}, FPS: {fps:.1f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/random_rollouts")
    args = parser.parse_args()
    
    collect_data(args.episodes, args.out)
