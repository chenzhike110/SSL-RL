from TD3 import TD3
import numpy as np
from actionmodule import ActionModule
from visionmodule import VisionModule

def angle_normalize(angle):
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi
        return angle

def get_state(robot_state, ball_state, action):
    robot_X = robot_state[0]
    robot_Y = robot_state[1]
    robot_orientation = robot_state[2]
    ball_X = ball_state[0]
    ball_Y = ball_state[1]
    robot_vx = action[0]
    robot_vy = action[1]

    robot2ball_theta = np.arctan2((ball_Y - robot_Y), (ball_X - robot_X))
    delt_theta = angle_normalize(robot2ball_theta - robot_orientation)
    robot2ball_dist = np.sqrt((robot_X-ball_Y)**2 + (robot_Y-ball_X)**2)
    robot_v_theta = np.arctan2(robot_vy, robot_vx)
    v_theta = angle_normalize(robot_v_theta + robot_orientation)
    v2ball_theta = angle_normalize(robot2ball_theta - v_theta)
    v = np.sqrt(robot_vx*robot_vx + robot_vy*robot_vy)
    v_t = v * np.cos(v2ball_theta)
    v_n = - v * np.sin(v2ball_theta)

    return [robot2ball_dist, np.sin(delt_theta), np.cos(delt_theta), v_t, v_n, np.sin(robot2ball_theta), np.cos(robot2ball_theta)]


if __name__ == "__main__":
    MULTI_GROUP = '224.5.23.2'
    VISION_PORT = 10094
    ACTION_IP = '127.0.0.1' # local host grSim
    ACTION_PORT = 20011 # grSim command listen port
    ROBOT_ID = 6
    vision = VisionModule(MULTI_GROUP, VISION_PORT)
    actionSender = ActionModule(ACTION_IP, ACTION_PORT)
    # actionSender.reset(robot_num=ROBOT_ID)
    state_dim = 7
    action_dim = 3
    max_action = np.array([1, 1, 1])
    min_action = np.array([-1, -1, -1])
    ACTION_BOUND = [min_action, max_action]
    max_v = 3.0

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action)
    file_name = "TD3_%s_%s" % ("RoboCup-v1", str(0))
    try:
        policy.load(filename=file_name, directory="./pytorch_models")
        print('Load model successfully !')
    except:
        print('WARNING: No model to load !')

    done = False
    vision.get_info(ROBOT_ID)
    obs = get_state(vision.robot_info, vision.ball_info, [0,0,0])
    while not done:
        action = policy.select_action(np.array(obs))
        action = np.clip(action, *ACTION_BOUND)
        actionSender.send_action(robot_num=ROBOT_ID, vx=action[0], vy=action[1], w=action[2])
        vision.get_info(ROBOT_ID)
        obs = get_state(vision.robot_info, vision.ball_info, action)
        