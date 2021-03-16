"""
@version: 2.0.0
@brief: 3 DoF version
@action dim: 3
@state dim: 7
@change log:
2019.2.24: correct observation_space dim
"""
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class RoboCupEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        # time step
        self.dt = 1.0 / 30
        # robot ability
        self.max_v = 3.0
        self.max_w = 2*np.pi
        self.robot_max_acc = 3
        self.robot_max_w_acc = 2*np.pi
        # robot ability for each step
        self.max_step_acc = self.robot_max_acc * self.dt
        self.max_step_w_acc = self.robot_max_w_acc * self.dt
        # robot action space
        self.low_action = np.array([-1, -1, -1])
        self.high_action = np.array([1, 1, 1])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)
        # robot observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # half field length and width
        self.field_length = 4.5
        self.field_width = 3
        # robot radius
        self.robot_r = 0.09
        # robot position
        self.x = 0.0
        self.y = 0.0
        # robot orientation
        self.orientation = 0.0
        # robot velocity angle in global coordinate system
        self.v_theta = 0.0
        # robot velocity angle in robot coordinate system
        self.robot_v_theta = 0.0
        # robot velocity
        self.v = 0.0
        # robot velocity in global coordinate system
        self.vx = 0.0
        self.vy = 0.0
        # robot velocity in robot coordinate system
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        # robot angular velocity
        self.w = 0.0
        # ball position
        self.ball_x = 0.0
        self.ball_y = 0.0

        # important variable
        self.robot2ball_dist = np.sqrt((self.x-self.ball_x)**2 + (self.y-self.ball_y)**2)
        self.robot2ball_theta = np.arctan2((self.ball_y - self.y), (self.ball_x - self.x))
        self.delt_theta = self.robot2ball_theta - self.orientation
        
        self.last_delt_theta = self.delt_theta
        self.last_robot2ball_dist = self.robot2ball_dist
        
        self.v2ball_theta = 0.0
        self.v_t = 0.0
        self.v_n = 0.0

        # environment feedback infomation
        self.state = [self.robot2ball_dist, np.sin(self.delt_theta), np.cos(self.delt_theta)]
        self.done = False
        self.reward = 0.0
        self.viewer = None
        
        self.seed()
        self.reset()

    """ action is [vx, vy, w. kick] """
    def step(self, action=[0,0,0,0]):
        # transfer from normalized action to normal action
        dvx = self.max_v*action[0] - self.robot_vx
        dvy = self.max_v*action[1] - self.robot_vy
        dw = self.max_w*action[2] - self.w

        # calculate step acceleration
        dv = np.sqrt(dvx*dvx + dvy*dvy)
        theta_acc = np.arctan2(dvy, dvx)
        # limit step acceleration
        if abs(dv) > self.max_step_acc:
            dv = np.clip(dv, -self.max_step_acc, self.max_step_acc)
            dvx = dv * np.cos(theta_acc)
            dvy = dv * np.sin(theta_acc)

        # update robot velocity in robot coordinate system
        self.robot_vx += dvx
        self.robot_vy += dvy

        # limit robot velocity
        self.robot_vx = np.clip(self.robot_vx, -self.max_v, self.max_v)
        self.robot_vy = np.clip(self.robot_vy, -self.max_v, self.max_v)
        self.v = np.sqrt(self.robot_vx*self.robot_vx + self.robot_vy*self.robot_vy)
        # update robot velocity angle
        self.robot_v_theta = np.arctan2(self.robot_vy, self.robot_vx)
        self.v_theta = self.robot_v_theta + self.orientation
        self.v_theta = self._angle_normalize(self.v_theta)
        
        # update robot angular velocity
        dw = np.clip(dw, -self.max_step_w_acc, self.max_step_w_acc)
        self.w += dw
        self.w = np.clip(self.w, -self.max_w, self.max_w)

        """
        ##################################################
        self.robot_v_theta = np.arctan2(self.robot_vy, self.robot_vx) + 0.5*self.w*self.dt
        self.v = np.sqrt(self.robot_vx**2 + self.robot_vy**2)
        self.robot_vx = self.v * np.cos(self.robot_v_theta)
        self.robot_vy = self.v * np.sin(self.robot_v_theta)
        self.v_theta = self.robot_v_theta + self.orientation
        self.v_theta = self._angle_normalize(self.v_theta)
        ##################################################
        """

        ##################### Dynamic Process ##########################

        # update the robot position according to the kinematic equation
        if(abs(self.w) < 0.001):
            self.x = self.x + self.v*np.cos(self.v_theta)*self.dt
            self.y = self.y + self.v*np.sin(self.v_theta)*self.dt
        else:
            self.x = self.x - (self.v/self.w)*np.sin(self.v_theta) + (self.v/self.w)*np.sin(self.v_theta + self.w * self.dt)
            self.y = self.y + (self.v/self.w)*np.cos(self.v_theta) - (self.v/self.w)*np.cos(self.v_theta + self.w * self.dt)
        
        # assume that robot velocity angle in robot coordinate system won't change
        self.v_theta = self.v_theta + self.w * self.dt
        self.v_theta = self._angle_normalize(self.v_theta)
        self.orientation = self.v_theta - self.robot_v_theta
        #self.orientation += self.w * self.dt
        #self.orientation = self._angle_normalize(self.orientation)
        #self.v_theta = self.orientation + self.robot_v_theta
        
        # store last info
        self.last_robot2ball_dist = self.robot2ball_dist
        self.last_delt_theta = self.delt_theta
        
        # calculate crucial variable
        self.robot2ball_dist = np.sqrt((self.x-self.ball_x)**2 + (self.y-self.ball_y)**2)
        self.robot2ball_theta = np.arctan2((self.ball_y - self.y), (self.ball_x - self.x))
        self.delt_theta = self.robot2ball_theta - self.orientation
        self.delt_theta = self._angle_normalize(self.delt_theta)
        
        self.v2ball_theta = self.robot2ball_theta - self.v_theta
        self.v2ball_theta = self._angle_normalize(self.v2ball_theta)
        self.v_t = self.v * np.cos(self.v2ball_theta)
        self.v_n = - self.v * np.sin(self.v2ball_theta)
        
        # judge the robot out of the field
        if self.x < -self.field_length or self.x > self.field_length or self.y <-self.field_width or self.y > self.field_width :
           self.done = True
        if self.robot2ball_dist < self.robot_r/3:
            self.done = True
            r_on_dribble = 100.0
            #if abs(self.robot2ball_theta) < np.pi/6:
            #    r_on_dribble = 100.0
            #else:
            #    r_on_dribble = 0.0
        else:
            r_on_dribble = -0.01  
        
        #total reward
        self.reward = r_on_dribble
        
        self.state = [self.robot2ball_dist, np.sin(self.delt_theta), np.cos(self.delt_theta), self.v_t, self.v_n, np.sin(self.robot2ball_theta), np.cos(self.robot2ball_theta)]

        return self.state, self.reward, self.done, {'state':[self.x, self.y, self.ball_x, self.ball_y]}

    def reset(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.x = np.random.randint(-100*self.field_length, 100*self.field_length)/100
        self.y = np.random.randint(-100*self.field_width,  100*self.field_width)/100
        self.orientation = np.random.randint(-180,  180)/180.0 * np.pi
        self.ball_x = np.random.randint(-100*self.field_length, 100*self.field_length)/100
        self.ball_y = np.random.randint(-100*self.field_width,  100*self.field_width)/100
        
        self.v = 0.0
        self.w = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.v_theta = 0.0
        self.robot_v_theta = 0.0
        
        self.robot2ball_dist = np.sqrt((self.x-self.ball_x)**2 + (self.y-self.ball_y)**2)
        self.robot2ball_theta = np.arctan2((self.ball_y - self.y), (self.ball_x - self.x))
        self.delt_theta = self.robot2ball_theta - self.orientation
        self.delt_theta = self._angle_normalize(self.delt_theta)
        self.last_robot2ball_dist = self.robot2ball_dist
        self.last_delt_theta = self.delt_theta
        
        self.v2ball_theta = self.robot2ball_theta - self.v_theta
        self.v2ball_theta = self._angle_normalize(self.v2ball_theta)
        self.v_t = self.v * np.cos(self.v2ball_theta)
        self.v_n = - self.v * np.sin(self.v2ball_theta)

        self.state = [self.robot2ball_dist, np.sin(self.delt_theta), np.cos(self.delt_theta), self.v_t, self.v_n, np.sin(self.robot2ball_theta), np.cos(self.robot2ball_theta)]
        self.done = False
        self.reward = 0.0
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 700)
            line1 = rendering.Line((50, 50), (950, 50))
            line1.set_color(0, 0, 0)
            line2 = rendering.Line((50, 50), (50, 650))
            line2.set_color(0, 0, 0)
            line3 = rendering.Line((50, 650), (950, 650))
            line3.set_color(0, 0, 0)
            line4 = rendering.Line((950, 50), (950, 650))
            line4.set_color(0, 0, 0)
            self.viewer.add_geom(line1)
            self.viewer.add_geom(line2)
            self.viewer.add_geom(line3)
            self.viewer.add_geom(line4)
        self.ball_img = rendering.make_circle(3)
        self.ball_img.set_color(0,0,255)
        self.car_img = rendering.make_circle(8)
        self.car_img.set_color(255,0,0)
        circle_transform = rendering.Transform(translation=(self.ball_x*100+450, self.ball_y*100+300))
        self.ball_img.add_attr(circle_transform)
        car_trans = rendering.Transform(translation=(self.x*100+450, self.y*100+300))
        self.car_img.add_attr(car_trans)
        self.viewer.add_geom(self.car_img)
        self.viewer.add_geom(self.ball_img)
        return self.viewer.render(return_rgb_array=0)

    def _angle_normalize(self, angle):
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi
        return angle