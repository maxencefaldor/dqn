#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from copy import deepcopy
from itertools import count

from replay_memory import replay_buffer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNAgent:
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 gamma=0.99,
                 batch_size=32,
                 n_gradient_steps=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 buffer_size=1e6,
                 beta=None):
        
        self.device = device
        self.n_actions = n_actions
        self.network = network.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(network.parameters(), lr=lr)
        self.target_network = deepcopy(self.network).to(self.device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_gradient_steps = n_gradient_steps
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.beta = beta
        self.replay_buffer = replay_buffer.ReplayBuffer(buffer_size)
        self.step = 0
        
        if self.beta is None:
            self.beta = 1
            self.update_target_network = self.periodic_update_target_network
        elif self.beta > 0 and self.beta < 1:
            self.update_target_network = self.smooth_update_target_network
        elif self.beta >= 1:
            self.beta = int(self.beta)
            self.update_target_network = self.periodic_update_target_network
        else:
            raise ValueError('Beta should be a positive integer or a real number in (0,1). Got: {}'.format(beta))
    
    def exponentially_decaying_epsilon(self):
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(-1. * self.step / self.epsilon_decay)
    
    def linearly_decaying_epsilon(self):
        self.epsilon = max(self.epsilon_min, 1 - self.step*(1 - self.epsilon_min)/self.epsilon_decay)
    
    def epsilon_greedy_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return torch.argmax(self.network(torch.Tensor(state).to(self.device).unsqueeze(0))).item()
        else:
            return torch.tensor(random.randrange(self.n_actions), device=self.device, dtype=torch.long).item()
    
    def greedy_action(self, state):
        with torch.no_grad():
            return torch.argmax(self.network(torch.Tensor(state).to(self.device).unsqueeze(0))).item()
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        batch = [*zip(*batch)]
        
        state_batch = torch.stack(batch[0]).to(self.device)
        action_batch = torch.stack(batch[1]).to(self.device)
        reward_batch = torch.stack(batch[2]).to(self.device)
        next_state_batch = torch.stack(batch[3]).to(self.device)
        done_batch = torch.stack(batch[4]).to(self.device)
        
        state_action_values = self.network(state_batch).gather(1, action_batch)
        next_state_action_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_state_action_values = reward_batch + self.gamma * next_state_action_values * (1 - done_batch)
        
        loss = self.criterion(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()
    
    def periodic_update_target_network(self):
        if self.step % self.beta == 0:
            self.target_network.load_state_dict(self.network.state_dict())  
    
    def smooth_update_target_network(self):
        for param_network, param_target_network in zip(self.network.parameters(), self.target_network.parameters()):
            param_target_network.data.copy_(param_network * self.beta + param_target_network * (1 - self.beta))
    
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
                else:
                    return
    
    def save(self, path):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path, map_location='cpu'):
        self.dqn.load_state_dict(torch.load(path, map_location=map_location))