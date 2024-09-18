from gymnasium.envs.registration import register

register(
    id = "environments/GridField-v0",
    entry_point="environments.envs:GridWorldEnv",
    max_episode_steps=100,
)