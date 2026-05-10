from gymnasium.envs.registration import register

register(
    id="MissileDefense-v0",
    entry_point="missile_defense.env:MissileDefenseEnv",
    max_episode_steps=3000,
)
