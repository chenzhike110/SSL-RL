# -*- coding: utf-8 -*-
import numpy as np
import gym
import ssl_env
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir="log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_v = 3.0
max_w = 2*np.pi
robot_max_acc = 3
robot_max_w_acc = 2*np.pi
l_max = (max_v)*(max_v)/(2*robot_max_acc)
dt = 1.0 / 30
max_step_acc = robot_max_acc * dt

def target_policy(state):
    theta = np.arctan2(state[1], state[2])

    if state[0] > l_max:
        v_t = max_v
    else:
        v_t = np.sqrt(2 * robot_max_acc * state[0])
        
    if state[4] > 0:
        v_n = state[4] - max_step_acc
        v_n = np.clip(v_n, 0, max_v)
    else:
        v_n = state[4] + max_step_acc
        v_n = np.clip(v_n, -max_v, 0)
        
    if theta > 0:
        w = np.sqrt(2 * robot_max_w_acc * theta)
    else:
        w = -np.sqrt(-2 * robot_max_w_acc * theta)
        
    """ mode 2: turn first
    if abs(theta) > np.pi/2:
        v_t = 0 
        v_t = 0
    else:
        v_t = v_t * np.exp(-0.75*(abs(theta)/(np.pi/6)))
        v_n = v_n * np.exp(-0.75*(abs(theta)/(np.pi/6)))
    """  
    vx = v_t*state[2] - v_n*state[1]
    vy = v_t*state[1] + v_n*state[2]
    return [vx/4.5, vy/4.5, w/(2*np.pi)]

def mimic(target_action, action):
    dist = 0
    for i in range(len(action)):
        dist += np.power(target_action[i] - action[i], 2)
    return np.exp(-2*dist)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        
# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
    
    def new_sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        sampled = np.asarray(self.storage)[ind]
        x, y, u, r, d = map(np.asarray, zip(*sampled))
        return x, y, u, r.reshape(-1,1), d.reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = torch.FloatTensor([max_action]).to(device)
        
        self.apply(weights_init)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = x * self.max_action
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        
        self.apply(weights_init)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.leaky_relu(self.l1(xu))
        x1 = F.leaky_relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.leaky_relu(self.l4(xu))
        x2 = F.leaky_relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.leaky_relu(self.l1(xu))
        x1 = F.leaky_relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1 

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())		

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        global total_timesteps
        for it in range(iterations):
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.new_sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)
            
            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(float(-self.max_action[0]), float(self.max_action[0]))

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            writer.add_scalar('max_Q', max(max(current_Q1),max(current_Q2)), total_timesteps)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            writer.add_scalar('critic_loss', critic_loss, total_timesteps)
            writer.add_scalar('actor_loss', actor_loss, total_timesteps)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pkl' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pkl' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pkl' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pkl' % (directory, filename)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="RoboCup-v1")			# OpenAI gym environment name
    parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
    #parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
    #parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
    parser.add_argument("--max_expl_noise", default=2.0, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--min_expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
    parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
    parser.add_argument("--load_model", default=False, type=bool)
    args = parser.parse_args()

    file_name = "TD3_%s_%s" % (args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
	
    state_dim = 7
    action_dim = 3
    max_action = np.array([1, 1, 1])
    min_action = np.array([-1, -1, -1])
    ACTION_BOUND = [min_action, max_action]
    VAR_MIN = args.min_expl_noise
    var = args.max_expl_noise

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action)
    if args.load_model:
        try:
            policy.load(filename=file_name, directory="./pytorch_models")
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')

    replay_buffer = ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = True
    reward = 0
    episode_reward = 0
    episode_timesteps = 0
    success = []

    while total_timesteps < args.max_timesteps:
        if done:
            # print success rate
            if reward > 90:
                success.append(1)
            else: success.append(0)
            if episode_num % 20 == 0 and episode_num >= 100:
                print('Successful Rate: ', sum(success[-100:]), '%')
                writer.add_scalar('success_rate', sum(success[-100:]), episode_num)
            if episode_num != 0:
                writer.add_scalar('episode_reward', episode_reward, episode_num)
            
            # Reset environment
            obs = env.reset()
            
            if total_timesteps != 0:
                #print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
            
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
            if episode_num % 50 == 0:
                policy.save("%s" % (file_name), directory="./pytorch_models")
                print('Model saved !')
                
        action = policy.select_action(np.array(obs))
        target_action = target_policy(obs)
        #if args.expl_noise != 0: 
        #    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
        action = np.clip(np.random.normal(action, var), *ACTION_BOUND)
        if total_timesteps>10000:
            var = max([var*.9999, VAR_MIN])

        # Perform action
        temp = env.step(action)
        new_obs, reward, done, _ = temp
        if reward > 90:
            print('hit the ball')
        mimic_reward = mimic(target_action, action) - 1
        writer.add_scalar('mimic_reward', mimic_reward, total_timesteps)
        reward += mimic_reward
        # env.render()
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
	
    #end of while 
    policy.save("%s" % (file_name), directory="./pytorch_models")