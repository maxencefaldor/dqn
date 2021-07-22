#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from copy import deepcopy
from itertools import count, product
import numpy as np
import matplotlib.pyplot as plt

from replay_memory import replay_buffer
from replay_memory import prioritized_replay_buffer
import gym
from gym import logger
logger.set_level(gym.logger.DISABLED)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
    
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
                return torch.argmax(self.dqn(torch.Tensor(state))).item()
        else:
            return torch.tensor(random.randrange(self.n_actions), device=device, dtype=torch.long).item()
    
    def greedy_action(self, state):
        with torch.no_grad():
            return torch.argmax(self.dqn(torch.Tensor(state))).item()
    
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
        
        state_batch = torch.stack(batch[0])
        action_batch = torch.stack(batch[1])
        reward_batch = torch.stack(batch[2])
        next_state_batch = torch.stack(batch[3])
        done_batch = torch.stack(batch[4])
        
        state_action_values = self.dqn(state_batch).gather(1, action_batch)
        next_state_action_values = self.update_rule(next_state_batch)
        expected_state_action_values = reward_batch + self.gamma * next_state_action_values * (1 - done_batch)
        
        if self.per:
            errors = (state_action_values - expected_state_action_values).data.numpy()
            for i, index in enumerate(indices):
                self.replay_buffer.update(index, errors[i])
            loss = self.criterion(state_action_values, torch.tensor(is_weights, dtype=torch.float32)*expected_state_action_values)
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
                    return_list.append(episode_return)
                    print("Episode {:4d} : {:4d} steps | epsilon = {:4.2f} | return = {:.1f}".format(i_episode, t+1, self.epsilon, episode_return))
                    break
        return return_list
    
    def test(self, env, step_max):
        episode_return = 0
        state = env.reset()
        for t in count():
            env.render()
            action = self.greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            state = next_state

            if done or t+1 >= step_max:
                env.close()
                return episode_return
    
    def save(self, name):
        torch.save(self.dqn.state_dict(), "trained_dqn/{}.pt".format(name))
    
    def load(self, name):
        self.dqn.load_state_dict(torch.load("trained_dqn/{}.pt".format(name)))


if __name__ == "__main__":
    class CartPoleSwingUp(gym.Wrapper):
        def __init__(self, env, **kwargs):
            super(CartPoleSwingUp, self).__init__(env, **kwargs)
            self.theta_dot_threshold = 4*np.pi
    
        def reset(self):
            self.env.env.state = [0, 0, np.pi, 0] + super().reset()
            self.env.env.steps_beyond_done = None
            return np.array(self.env.env.state)
    
        def step(self, action):
            state, reward, done, _ = super().step(action)
            x, x_dot, theta, theta_dot = state
            
            done = x < -self.x_threshold \
                   or x > self.x_threshold \
                   or theta_dot < -self.theta_dot_threshold \
                   or theta_dot > self.theta_dot_threshold
            
            if done:
                # game over
                reward = -10.
                if self.steps_beyond_done is None:
                    self.steps_beyond_done = 0
                else:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                    self.steps_beyond_done += 1
            else:
                if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                    # pole upright
                    reward = 1.
                else:
                    # pole swinging
                    reward = 0.
    
            return np.array(self.state), reward, done, {}
    
    env = CartPoleSwingUp(gym.make('CartPole-v1'))
    
    dqn_1 = DQN()
    dqn_2 = deepcopy(dqn_1)
    dqn_3 = deepcopy(dqn_1)
    dqn_4 = deepcopy(dqn_1)
    
    agent_1 = DQNAgent(2,
                     dqn_1,
                     lr=0.001,
                     gamma=0.99,
                     batch_size=32,
                     n_gradient_steps=1,
                     epsilon_min=0.01,
                     epsilon_decay=2000,
                     per=True,
                     buffer_size=1e6,
                     beta=100,
                     option=0)
    agent_2 = DQNAgent(2,
                     dqn_2,
                     lr=0.001,
                     gamma=0.99,
                     batch_size=32,
                     n_gradient_steps=1,
                     epsilon_min=0.01,
                     epsilon_decay=2000,
                     per=False,
                     buffer_size=1e6,
                     beta=100,
                     option=0)
    agent_3 = DQNAgent(2,
                     dqn_3,
                     lr=0.001,
                     gamma=0.99,
                     batch_size=32,
                     n_gradient_steps=1,
                     epsilon_min=0.01,
                     epsilon_decay=2000,
                     per=True,
                     buffer_size=1e6,
                     beta=100,
                     option=2)
    agent_4 = DQNAgent(2,
                     dqn_4,
                     lr=0.001,
                     gamma=0.99,
                     batch_size=32,
                     n_gradient_steps=1,
                     epsilon_min=0.01,
                     epsilon_decay=2000,
                     per=False,
                     buffer_size=1e6,
                     beta=100,
                     option=2)
    return_list_1 = agent_1.train(env, 200)
    return_list_2 = agent_2.train(env, 200)
    return_list_3 = agent_3.train(env, 200)
    return_list_4 = agent_4.train(env, 200)
    
    plt.plot(return_list_1, label="Agent 1")
    plt.plot(return_list_2, label="Agent 2")
    plt.plot(return_list_3, label="Agent 3")
    plt.plot(return_list_4, label="Agent 4")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.legend()
    plt.savefig('plot_2.png', dpi=1200)
    plt.show()
    
    agent_1.test(env, 1000)
    agent_2.test(env, 1000)
    agent_3.test(env, 1000)
    agent_4.test(env, 1000)
