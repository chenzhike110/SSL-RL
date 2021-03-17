"""
@version: 2.0.0
@brief: 3 DoF version
@action dim: 3
@state dim: 7
@change log:
2019.2.24: correct observation_space dim
"""
from os import stat
from gym.logger import info
from TD3 import done, reward
from test_grSim import actionSender, vision
import gym
from gym import spaces
from actionmodule import ActionModule
from visionmodule import VisionModule
import numpy as np

class GrsimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, ROBOT_ID, train, targetPos=None):
        self.MULTI_GROUP = '224.5.23.2'
        self.VISION_PORT = 10094
        self.ACTION_IP = '127.0.0.1' # local host grSim
        self.ACTION_PORT = 20011 # grSim command listen port
        self.ROBOT_ID = ROBOT_ID
        self.vision = VisionModule(self.MULTI_GROUP, self.VISION_PORT)
        self.actionSender = ActionModule(self.ACTION_IP, self.ACTION_PORT)
        self.train = train
        self.lowest = [-3.5, -3.5, -3]
        self.uppest = [3.5, 3.5, 3]
        self.height = 3.0
        self.width = 4.5
        self.action_space = spaces.Box(low=self.lowest, high=self.uppest, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.targetPos = 

    def in_range(self, state):
        if state[0] < -self.width or state[0] > self.width:
            return False
        if state[1] < -self.height or state[1] > self.height:
            return False
        return True

    """ action is [vx, vy, w. kick] """
    def step(self, action=[0,0,0,0]):
        reward = -0.1
        if action[3] > 0 and self.vision.robot_info[3]!=1:
            action[3] = 0
            reward -= 0.5
        self.actionSender.send_action(robot_num=self.ROBOT_ID, vx=action[0], vy=action[1], w=action[2], kp=action[3])
        self.vision.get_info(self.ROBOT_ID)
        state = [self.vision.robot_info, self.vision.ball_info]
        done =  False
        '''
            kick ball task
        '''
        if self.vision.robot_info[3] == 1:
            done = True
            reward = self.get_reward()
            # if 
        # if not self.in_range(state[0]) or not self.in_range(state[1]):
        #     done = True
        '''

        '''
        return state, reward, done, info
    
    def get_reward(self):
        self.vision.get_info(self.ROBOT_ID)



    def reset(self):
        if self.train:
            actionSender.send_reset(self.ROBOT_ID, 1)
        else:
            actionSender.send_reset(self.ROBOT_ID, 0)
        self.vision.get_info(self.ROBOT_ID)
        state = [self.vision.robot_info, self.vision.ball_info[:2]]
        return state

if __name__ == "__main__":
    env = GrsimEnv(6, True)
    observate = env.reset()

