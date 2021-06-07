from gym.envs.registration import register

register(
    id='deepwatch-v0',
    entry_point='deepwatch.envs:DeepwatchEnv',
)