#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a DQN agent."""

import math
import random
from copy import deepcopy
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory.replay_buffer import ReplayBuffer
from replay_memory.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNAgent(object):
    """Implementation of a DQN agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 gamma=0.99,
                 n=1,
                 n_gradient_steps=1,
                 beta=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 batch_size=32,
                 per=False,
                 buffer_size=1e6):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take.
            network: `torch.nn`, neural network used to approximate Q.
            lr: float, learning rate.
            gamma: float, discount rate.
            n: int, number of steps of bootstrapping.
            batch_size: int, batch size.
            n_gradient_steps: int, number of gradient steps taken during a
                time step.
            epsilon_min: float, the minimum epsilon value during training.
            epsilon_decay: int, epsilon decay parameter.
            buffer_size: int, capacity of the replay buffer.
            beta: float, update period for the target network if beta
                is a positive integer. Soft update parameter for the target
                network if beta is a float in (0, 1).
        """
        self._device = device
        self.n_actions = n_actions
        self.network = network.to(self._device)
        self.target_network = deepcopy(self.network).to(self._device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr)
        self.gamma_n = math.pow(gamma, n)
        self.n = n
        self.n_gradient_steps = n_gradient_steps
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.per = per
        
        if beta > 0 and beta < 1:
            self.beta = beta
            self._update_target_network = self._soft_update_target_network
        elif beta >= 1 and (isinstance(beta, int) or beta.is_integer()):
            self.beta = int(beta)
            self._update_target_network = self._hard_update_target_network
        else:
            raise ValueError("Beta should be a positive integer or a real "
                             "number in (0, 1). Got: {}".format(beta))
        
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(gamma=gamma,
                                                         n=self.n,
                                                         buffer_size=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(gamma=gamma,
                                              n=self.n,
                                              buffer_size=buffer_size)
        
        self.step = 0
    
    def _linearly_decaying_epsilon(self):
        """Set epsilon for the epsilon-greedy policy.
        
        Linearly decay epsilon from 1. to epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, 1 - self.step*(1 - self.epsilon_min)/self.epsilon_decay)
    
    def _exponentially_decaying_epsilon(self):
        """Set epsilon for the epsilon-greedy policy.
        
        Exponentially decay epsilon from 1. to epsilon_min.
        """
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(-1. * self.step / self.epsilon_decay)
    
    def greedy_action(self, state):
        """Returns an action following a greedy policy.
        
        Args:
            state: torch.Tensor, state of the agent.
        
        Returns:
            int, greedy action.
        """
        with torch.no_grad():
            return torch.argmax(self.network(torch.Tensor(state).to(self._device).unsqueeze(0))).item()
    
    def epsilon_greedy_action(self, state):
        """Returns an action following an epsilon-greedy policy.
        
        Args:
            state: torch.Tensor, state of the agent.
        
        Returns:
            int, epsilon-greedy action.
        """
        if random.random() > self.epsilon:
            return self.greedy_action(state)
        else:
            return random.randint(0, self.n_actions - 1)
    
    def _hard_update_target_network(self):
        """Periodically update the target network."""
        if self.step % self.beta == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def _soft_update_target_network(self):
        """Soft update the target network"""
        for param_network, param_target_network in zip(self.network.parameters(), self.target_network.parameters()):
            param_target_network.data.copy_(param_network * self.beta + param_target_network * (1 - self.beta))
    
    def _next_state_q(self, next_state_batch):
        """Returns the next_state Q-values
        
        Args:
            next_state_batch: tuple, batch of next state.
        
        Returns:
            torch.Tensor, Q-values of the batch.
        """
        return self.target_network(
            next_state_batch).max(1)[0].unsqueeze(1).detach()
    
    def learn(self):
        """Learns the Q-value from the replay memory."""
        if len(self.replay_buffer) - self.n + 1 < self.batch_size:
            return
        
        if self.per:
            indices, batch, is_weights = self.replay_buffer.sample(self.batch_size)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = torch.stack(batch.state).to(self._device)
        action_batch = torch.stack(batch.action).to(self._device)
        reward_batch = torch.stack(batch.reward).to(self._device)
        next_state_batch = torch.stack(batch.next_state).to(self._device)
        done_batch = torch.stack(batch.done).to(self._device)
        
        state_action_values = self.network(state_batch).gather(1, action_batch)
        next_state_action_values = self._next_state_q(next_state_batch)
        expected_state_action_values = reward_batch + self.gamma_n * next_state_action_values * (1 - done_batch)
        
        if self.per:
            errors = state_action_values - expected_state_action_values
            for i, index in enumerate(indices):
                self.replay_buffer.update(index, errors[i][0].item())
            loss = F.mse_loss(state_action_values,
                              expected_state_action_values,
                              reduction='none')
            loss *= torch.tensor(is_weights.reshape(-1, 1),
                                 dtype=torch.float32).to(self._device)
            loss = loss.mean()
        else:
            loss = self.criterion(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_network()
    
    def train(self, env, n_episodes):
        """Trains the Q-network.
        
        Args:
            env: gym.env, Gym environment.
            n_episodes: int, number of episodes to train for.
        
        Returns:
            list of float, list of episode's return.
        """
        return_list = []
        for i_episode in range(1, n_episodes+1):
            episode_return = 0
            state = env.reset()
            for t in count():
                self._linearly_decaying_epsilon()
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.replay_buffer.add(torch.tensor(state, dtype=torch.float32),
                                       torch.tensor([action], dtype=torch.long),
                                       reward,
                                       torch.tensor(next_state, dtype=torch.float32),
                                       done)
                state = next_state
                episode_return += reward
                self.step += 1
                
                for _ in range(self.n_gradient_steps):
                    self.learn()
                
                if done:
                    return_list.append(episode_return)
                    print("Episode {:4d} : {:4d} steps | epsilon = {:4.2f} | return = {:.1f}"
                          .format(i_episode, t+1, self.epsilon, episode_return))
                    
                    if return_list and episode_return >= max(return_list):
                        self.save("model.pt")
                    break
            
        return return_list
    
    def test(self, env, step_max=np.inf, render_video=True):
        """Tests the agent in the environment.
        
        Args:
            env: gym.env, Gym environment.
            step_max: int, maximum number of steps
            render_video: bool, if True, create a video instead of rendering.
        """
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
        """Saves the Q-network to a disk file.
        
        Args:
            path: str, path of the disk file.
        """
        torch.save(self.network.state_dict(), path)
    
    def load(self, path, map_location='cpu'):
        """Loads a saved Q-network from a disk file.
        
        Args:
            path: str, path of the disk file.
            map_location: str, string specifying how to remap storage locations
        """
        self.network.load_state_dict(torch.load(path, map_location=map_location))
