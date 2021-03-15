from gym.envs.registration import register

register(
    id='RoboCup-v1',
    entry_point='ssl_env.ssl_env:RoboCupEnv',
    max_episode_steps=500,
    reward_threshold=500,
)
