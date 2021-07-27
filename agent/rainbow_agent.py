#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import base64
import shutil
import math
import random
from copy import deepcopy
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import HTML

from replay_memory import replay_buffer
from replay_memory import prioritized_replay_buffer
import gym
from gym import logger
logger.set_level(gym.logger.DISABLED)
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from gym.wrappers import Monitor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self,
                 n_actions,
                 dqn,
                 lr=0.001,
                 gamma=0.99,
                 batch_size=32,
                 n_gradient_steps=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 per=False,
                 buffer_size=1e6,
                 beta=None,
                 option=0):
        
        self.n_actions = n_actions
        self.dqn = dqn.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(dqn.parameters(), lr=lr)
        self.target_dqn = deepcopy(self.dqn).to(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_gradient_steps = n_gradient_steps
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.per = per
        self.beta = beta
        self.option = option
        self.step = 0
        
        if self.per:
            self.replay_buffer = prioritized_replay_buffer.ReplayBuffer(buffer_size)
        else:
            self.replay_buffer = replay_buffer.ReplayBuffer(buffer_size)
        
        if self.beta is None:
            self.update_target_dqn = (lambda: None)
        elif isinstance(self.beta, int):
            self.update_target_dqn = self.periodic_update_target_dqn
        elif self.beta > 0 and self.beta < 1:
            self.update_target_dqn = self.smooth_update_target_dqn
        else:
            raise ValueError('Beta should be a positive integer or a real number in (0,1). Got: {}'.format(beta))
        
        if self.option == 0:
            self.update_rule = self.vanilla_update
        elif self.option == 1:
            self.update_rule = self.target_update
        elif self.option == 2:
            self.update_rule = self.double_q_learning_update
        else:
            raise ValueError('Option should be 0, 1 or 2. Got: {}'.format(option))
    
    def exponentially_decaying_epsilon(self):
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(-1. * self.step / self.epsilon_decay)
    
    def linearly_decaying_epsilon(self):
        self.epsilon = max(self.epsilon_min, 1 - self.step*(1 - self.epsilon_min)/self.epsilon_decay)
    
    def epsilon_greedy_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return torch.argmax(self.dqn(torch.Tensor(state).to(device).unsqueeze(0))).item()
        else:
            return torch.tensor(random.randrange(self.n_actions), device=device, dtype=torch.long).item()
    
    def greedy_action(self, state):
        with torch.no_grad():
            return torch.argmax(self.dqn(torch.Tensor(state).to(device).unsqueeze(0))).item()
    
    def vanilla_update(self, next_state_batch):
        return self.dqn(next_state_batch).max(1)[0].unsqueeze(1).detach()
    
    def target_update(self, next_state_batch):
        return self.target_dqn(next_state_batch).max(1)[0].unsqueeze(1).detach()
    
    def double_q_learning_update(self, next_state_batch):
        return self.target_dqn(next_state_batch).gather(1, self.dqn(next_state_batch).max(1)[1].unsqueeze(1)).detach()
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.per:
            indices, batch, is_weights = self.replay_buffer.sample(self.batch_size)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
        batch = [*zip(*batch)]
        
        state_batch = torch.stack(batch[0]).to(device)
        action_batch = torch.stack(batch[1]).to(device)
        reward_batch = torch.stack(batch[2]).to(device)
        next_state_batch = torch.stack(batch[3]).to(device)
        done_batch = torch.stack(batch[4]).to(device)
        
        state_action_values = self.dqn(state_batch).gather(1, action_batch)
        next_state_action_values = self.update_rule(next_state_batch)
        expected_state_action_values = reward_batch + self.gamma * next_state_action_values * (1 - done_batch)
        
        if self.per:
            errors = state_action_values - expected_state_action_values
            for i, index in enumerate(indices):
                self.replay_buffer.update(index, errors[i][0].item())
            loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='none')
            loss *= torch.tensor(is_weights.reshape(-1, 1), dtype=torch.float32).to(device)
            loss = loss.mean()
        else:
            loss = self.criterion(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_dqn()
    
    def periodic_update_target_dqn(self):
        if self.step % self.beta == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())  
    
    def smooth_update_target_dqn(self):
        for param_dqn, param_target_dqn in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            param_target_dqn.data.copy_(param_dqn * self.beta + param_target_dqn * (1 - self.beta))
    
    def train(self, env, n_episodes):
        return_list = []
        for i_episode in range(1, n_episodes+1):
            episode_return = 0
            state = env.reset()
            for t in count():
                self.linearly_decaying_epsilon()
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_return += reward
                
                self.replay_buffer.add(torch.Tensor(state),
                                       torch.tensor([action], dtype=torch.long),
                                       torch.tensor([reward], dtype=torch.float),
                                       torch.Tensor(next_state),
                                       torch.tensor([int(done)], dtype=torch.long))
                state = next_state
                self.step += 1
                
                for _ in range(self.n_gradient_steps):
                    self.learn()
                
                if done:
                    if return_list and episode_return >= max(return_list):
                        self.save("model.pt")
                    return_list.append(episode_return)
                    print("Episode {:4d} : {:4d} steps | epsilon = {:4.2f} | return = {:.1f}".format(i_episode, t+1, self.epsilon, episode_return))
                    break
        return return_list
    
    def test(self, env, step_max=np.inf, render_video=True):
        if render_video:
            env = Monitor(env, "./tmp", force=True)
        episode_return = 0
        state = env.reset()
        for t in count():
            if not render_video:
                env.render()
            action = self.greedy_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            state = next_state

            if done or t+1 >= step_max:
                env.close()
                if render_video:
                    video = io.open("./tmp/openaigym.video.{}.video000000.mp4".format(env.file_infix), 'r+b').read()
                    encoded = base64.b64encode(video)
                    return HTML(data='''
                        <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
                    .format(encoded.decode('ascii')))
                    shutil.rmtree("./tmp")
    
    def save(self, path):
        torch.save(self.dqn.state_dict(), path)
    
    def load(self, path, map_location='cpu'):
        self.dqn.load_state_dict(torch.load(path, map_location=map_location))


if __name__ == "__main__":
    import time
    from gym.wrappers import AtariPreprocessing
    from gym.wrappers import FrameStack
    
    class ActionWrapper(AtariPreprocessing):
        def __init__(self, env, **kwargs):
            super(ActionWrapper, self).__init__(env, **kwargs)
    
        def step(self, action):
            return super(ActionWrapper, self).step(4 + action)
    
    env = FrameStack(ActionWrapper(gym.make('PongNoFrameskip-v4'),
                             noop_max=0,
                             frame_skip=4,
                             terminal_on_life_loss=True,
                             grayscale_obs=True,
                             scale_obs=True), 4)
    
    class AtariCNN(nn.Module):
        def __init__(self, in_channels=4, n_actions=2):
            super(AtariCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(7 * 7 * 64, 512)
            self.fc2 = nn.Linear(512, n_actions)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            return self.fc2(x)
    
    dqn = AtariCNN()
    
    agent = DQNAgent(2,
                     dqn,
                     lr=0.001,
                     gamma=0.99,
                     batch_size=32,
                     n_gradient_steps=1,
                     epsilon_min=0.01,
                     epsilon_decay=6000,
                     per=True,
                     buffer_size=1e5,
                     beta=100,
                     option=2)
    agent.train(env, 25)
    