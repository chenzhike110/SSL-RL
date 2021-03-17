"""
@version: 2.0.0
@brief: 3 DoF version
@action dim: 3
@state dim: 7
@change log:
2019.2.24: correct observation_space dim
"""
from math import atan2
from os import stat
from random import random
from gym.logger import info
# from torch._C import Value
import gym
from gym import spaces
from actionmodule import ActionModule
from visionmodule import VisionModule
import numpy as np
import time

class GrsimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, ROBOT_ID, train, targetLine=None):
        self.MULTI_GROUP = '224.5.23.2'
        self.VISION_PORT = 10094
        self.ACTION_IP = '127.0.0.1' # local host grSim
        self.ACTION_PORT = 20011 # grSim command listen port
        self.ROBOT_ID = ROBOT_ID
        self.STATUS_PORT = 30011
        self.vision = VisionModule(self.MULTI_GROUP, self.VISION_PORT, self.STATUS_PORT)
        self.actionSender = ActionModule(self.ACTION_IP, self.ACTION_PORT)
        self.train = train
        self.lowest = np.array([-3])
        self.uppest = np.array([3])
        self.height = 3.0
        self.width = 4.5
        self.action_space = spaces.Box(low=self.lowest, high=self.uppest, dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.targetLine = targetLine

    def in_range(self, state):
        if state[0] < -self.width or state[0] > self.width:
            return False
        if state[1] < -self.height or state[1] > self.height:
            return False
        return True

    """ action is [w. kick] """
    def step(self, action=[0,0]):
        reward = -0.1
        if action[1] > 0 and self.vision.robot_info[3]!=1:
            done = True
            reward -= 0.5
        if action[1] == 1:
            kp = 5
        else:
            kp = 0
        self.actionSender.send_action(robot_num=self.ROBOT_ID, vx=0, vy=0, w=action[0], kp=kp)
        self.vision.get_info(self.ROBOT_ID)
        state = [self.vision.robot_info, self.vision.ball_info]
        done =  False
        '''
            kick ball task
        '''
        if action[1] == 1:
            done = True
            reward = self.get_reward()
            # if 
        # if not self.in_range(state[0]) or not self.in_range(state[1]):
        #     done = True
        '''

        '''
        return state, reward, done, info
    
    def get_reward(self):
        ball_pos = [self.vision.ball_info[0], self.vision.ball_info[1]]
        self.vision.get_info(self.ROBOT_ID)
        theta_1 = atan2(self.targetLine[0][1]-ball_pos[1], self.targetLine[0][0]-ball_pos[0])
        theta_2 = atan2(self.targetLine[1][1]-ball_pos[1], self.targetLine[1][0]-ball_pos[0])
        if self.in_range(atan2(self.vision.ball_info[3],self.vision.ball_info[2]), theta_1, theta_2):
            reward = 100
        else:
            reward = -100
        return reward

    def in_range(self, value, set_1, set_2):
        if value < set_1 and value > set_2:
            return True
        if value > set_1 and value < set_2:
            return True
        return False

    def reset(self):
        if self.train:
            self.actionSender.send_reset(self.ROBOT_ID, 1)
        else:
            self.actionSender.send_reset(self.ROBOT_ID, 0)
        self.vision.get_info(self.ROBOT_ID)
        state = [self.vision.robot_info, self.vision.ball_info[:2]]
        if state[0][3] != 1:
            self.reset()
        return state

if __name__ == "__main__":
    env = GrsimEnv(6, True, [[4.5, 0.5],[4.5, -0.5]])
    observate = env.reset()
    time.sleep(1)
    done = False
    while not done:
        action = env.action_space.sample()
        if random() < 0.99:
            action = np.append(action, 1)
        else:
            action = np.append(action, 1)
        state, reward, done, info = env.step(action)
        print(reward)

