from gymnasium.envs.registration import register

register(
    id='Deepwatch-v0',
    entry_point='Deepwatch.envs:DeepwatchEnv',
)