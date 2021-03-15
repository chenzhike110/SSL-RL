# -*- coding: utf-8 -*-
import numpy as np
import gym
import ssl_env
import argparse
import matplotlib.pyplot as plt

IS_TRAIN = False
IS_LOAD = False#True
IS_SAVE = False
USE_NOISE = False
USE_POLICY = False#True
PLOT_RESULTS = True
point_size = 5

max_v = 3.0
max_w = 2*np.pi
robot_max_acc = 3
robot_max_w_acc = 2*np.pi
l_max = (max_v)*(max_v)/(2*robot_max_acc)
dt = 1.0 / 30
max_step_acc = robot_max_acc * dt
        

def plot_episode(robot_x, robot_y, robot_x_neg, robot_y_neg, ball_x, ball_y, episode_num):
    static_x = []
    static_y = []
    static_x.append(-4.5)
    static_y.append(-3)
    static_x.append(4.5)
    static_y.append(3)
    static_x.append(-4.5)
    static_y.append(3)
    static_x.append(4.5)
    static_y.append(-3)
    plt.figure('fig', figsize=(10,10))
    plt.clf()
    ax = plt.gca()
    ax.set_aspect(1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(static_x, static_y, c='g', s=20, alpha=1.0)
    ax.scatter(ball_x, ball_y, c='r', s=100)
    ax.scatter(robot_x, robot_y, c='b', s=point_size, alpha=0.7)
    ax.scatter(robot_x_neg, robot_y_neg, c='r', s=point_size, alpha=1.0)
    # plt.savefig(str(episode_num)+'.png', dpi=600)
    # plt.show()
    
def plot_action(action_log, episode_num):
    vx = [item[0] for item in action_log]
    vy = [item[1] for item in action_log]
    w = [item[2] for item in action_log]
    t = [item for item in range(len(action_log))]
    plt.figure('fig', figsize=(10,10))
    plt.clf()
    ax = plt.gca()
    ax.set_xlabel('step')
    ax.set_ylabel('action')
    ax.plot(t, vx, 'r-')
    ax.plot(t, vy, 'g-')
    ax.plot(t, w, 'b-')
    # plt.savefig(str(episode_num)+'.act.png', dpi=600)

def target_policy(state):
    theta = np.arctan2(state[1], state[2])

    v_t = 0.0
    v_n = 0.0
    if state[0] > 1.5:
        v_t = max_v
    else:
        v_t = np.sqrt( 6 * state[0] )
    if state[4] > 0:
        v_n = state[4] - max_step_acc
        v_n = np.clip(v_n, 0, max_v)
    else:
        v_n = state[4] + max_step_acc
        v_n = np.clip(v_n, -max_v, 0)
        
    if theta > 0:
        w = np.sqrt(4*np.pi*theta)
    else:
        w = -np.sqrt(-4*np.pi*theta)
    w = 0
    vx = v_t*state[2] - v_n*state[1]
    vy = v_t*state[1] + v_n*state[2]
        
    return [vx/max_v, vy/max_v, w/max_w], [v_t, v_n, theta]
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="RoboCup-v1")			# OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
    parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for

    args = parser.parse_args()

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
	
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    total_timesteps = 0
    episode_num = 0
    done = True
    reward = 0
    episode_reward = 0
    episode_timesteps = 0
    success = []
    ball_x = 0
    ball_y = 0
    robot_x = []
    robot_y = []
    robot_x_neg = []
    robot_y_neg = []  
    action_log = []

    while total_timesteps < args.max_timesteps:
        if done:
            print(episode_reward)
            if episode_reward > 90:
                success.append(1)
            else: success.append(0)
            if episode_num % 100 == 0 and episode_num > 0:
                print('Successful Rate: ', sum(success[-100:]), '%')
                
            if PLOT_RESULTS:
                plot_episode(robot_x, robot_y, robot_x_neg, robot_y_neg, ball_x, ball_y ,episode_num)
                plot_action(action_log, episode_num)
                if episode_num > 500:
                    break
            # Reset environment
            obs = env.reset()
            robot_x = []
            robot_y = []
            robot_x_neg = []
            robot_y_neg = []
            action_log = []
            
            new_obs, reward, done, plot_info = env.step([0,0,0])
            ball_x = plot_info[2]
            ball_y = plot_info[3]
            
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        action, real_action = target_policy(obs)

        # Perform action
        new_obs, reward, done, plot_info = env.step(action)
        action_log.append(real_action)
        if action[0] <0:
            robot_x_neg.append(plot_info[0])
            robot_y_neg.append(plot_info[1])
        else:
            robot_x.append(plot_info[0])
            robot_y.append(plot_info[1])
        
        # env.render()
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1