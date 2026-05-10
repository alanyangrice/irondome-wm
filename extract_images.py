import numpy as np
import cv2
import os
import gymnasium as gym
import missile_defense

def extract_frames():
    os.makedirs("data/sample_images", exist_ok=True)
    
    # 1. Extract from NPZ
    if os.path.exists("data/random_rollouts/ep_0.npz"):
        data = np.load("data/random_rollouts/ep_0.npz")
        obs = data["obs"]
        # Save a few frames
        for i in range(0, min(50, len(obs)), 10):
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(obs[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/sample_images/obs_frame_{i}.png", img_bgr)
            
    # 2. Generate a human render frame
    env = gym.make("MissileDefense-v0", render_mode="human")
    env.reset()
    
    # Run a few steps to get missiles on screen
    for _ in range(30):
        env.step(env.action_space.sample())
        
    # Get the human render image
    img = env.unwrapped.renderer.render_human(
        env.unwrapped.turret, 
        env.unwrapped.missiles, 
        env.unwrapped.last_laser_fired, 
        env.unwrapped.explosions, 
        env.unwrapped.clouds,
        env.unwrapped.birds,
        env.unwrapped.score, 
        env.unwrapped.steps
    )
    
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("data/sample_images/human_render.png", img_bgr)
    env.close()

if __name__ == "__main__":
    extract_frames()
