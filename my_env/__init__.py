from gym.envs.registration import register

register(
    id='RoboCup-v2',
    entry_point='my_env.my_env:GrsimEnv',
    max_episode_steps=500,
    reward_threshold=500,
)
