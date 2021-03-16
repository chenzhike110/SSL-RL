import TD3
import numpy as np
import argparse
import gym
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="RoboCup-v1")			# OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()

    env = gym.make(args.env_name)
    state_dim = 7
    action_dim = 3
    max_action = np.array([1, 1, 1])
    min_action = np.array([-1, -1, -1])
    ACTION_BOUND = [min_action, max_action]

    # Initialize policy
    policy = TD3.TD3(state_dim, action_dim, max_action)
    file_name = "TD3_%s_%s" % (args.env_name, str(args.seed))
    try:
        policy.load(filename=file_name, directory="./pytorch_models")
        print('Load model successfully !')
    except:
        print('WARNING: No model to load !')

    done = False
    obs = env.reset()
    while not done:
        action = policy.select_action(np.array(obs))
        temp = env.step(action)
        new_obs, reward, done, _ = temp
        env.render()
        time.sleep(0.05)
        obs = new_obs
    while True:
        pass