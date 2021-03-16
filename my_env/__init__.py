from gym.envs.registration import register

register(
    id='RoboCup-v1',
    entry_point='my_env.my_env:RoboCupEnv',
    max_episode_steps=500,
    reward_threshold=500,
)
