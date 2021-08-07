#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a DQN agent.

The algorithm and default hyperparameters follow "Playing Atari with Deep
Reinforcement Learning", Mnih et al. (2013).

In addition to this, the agent can perform
    * prioritized experience replay
    * multi-step bootstrapping
"""

import os
import math
import random
from copy import deepcopy
from itertools import count
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_memory.replay_buffer import ReplayBuffer
from replay_memory.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNAgent(object):
    """Implementation of a DQN agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 criterion=nn.MSELoss,
                 gamma=0.99,
                 n=1,
                 n_gradient_steps=1,
                 beta=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 batch_size=32,
                 buffer_size=1e6,
                 per=False):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take at any state.
            network: `torch.nn`, neural network used to approximate the
                Q-value.
            lr: float, learning rate.
            criterion: `nn.modules.loss`, loss used to train the network.
            gamma: float, discount rate.
            n: int, number of steps of bootstrapping.
            n_gradient_steps: int, number of gradient steps taken by time step.
            beta: float, update period for the target network if beta
                is a positive integer. Soft update parameter for the target
                network if beta is a float in (0, 1).
            epsilon_min: float, the minimum epsilon value during training.
            epsilon_decay: int, epsilon decay parameter.
            batch_size: int, batch size.
            buffer_size: int, capacity of the replay buffer.
            per: bool, If True, use prioritized experience replay, else use
                uniformly sampled experience replay.
        """
        self._device = device
        self.n_actions = n_actions
        self.network = network.to(self._device)
        self.target_network = deepcopy(self.network).to(self._device)
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
            self.replay_buffer = PrioritizedReplayBuffer(device=self._device,
                                                         gamma=gamma,
                                                         n=self.n,
                                                         buffer_size=buffer_size)
            self.criterion = criterion(reduction='none')
        else:
            self.replay_buffer = ReplayBuffer(device=self._device,
                                              gamma=gamma,
                                              n=self.n,
                                              buffer_size=buffer_size)
            self.criterion = criterion()
        
        self.step = 0
    
    def _linearly_decay_epsilon(self):
        """Linearly decay epsilon from 1. to epsilon_min according to
        self.step for the epsilon-greedy policy.
        """
        self.epsilon = max(self.epsilon_min, 1 - self.step*(
            1 - self.epsilon_min)/self.epsilon_decay)
    
    def _exponentially_decay_epsilon(self):
        """Exponentially decay epsilon from 1. to epsilon_min according to
        self.step for the epsilon-greedy policy.
        """
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(
            -self.step/self.epsilon_decay)
    
    def greedy_action(self, state):
        """Returns an action following the greedy policy.
        
        Args:
            state: `torch.Tensor`, state of the agent.
        
        Returns:
            int, greedy action.
        """
        with torch.no_grad():
            return torch.argmax(self.network(torch.Tensor(state).to(self._device).unsqueeze(0))).item()
    
    def epsilon_greedy_action(self, state):
        """Returns an action following the epsilon-greedy policy.
        
        Args:
            state: `torch.Tensor`, state of the agent.
        
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
    
    def _next_state_q_values(self, next_states):
        """Returns the next state Q-values.
        
        Args:
            next_state_batch: `torch.Tensor`, batch of next state.
        
        Returns:
            `torch.Tensor`, next state Q-values.
        """
        return self.target_network(
            next_states).max(1)[0].unsqueeze(1).detach()
    
    def _target_state_q_values(self, rewards, next_states, dones):
        """Returns the target Q-values.
        
        Args:
            rewards: `torch.Tensor`, batch of multi-step returns.
            next_state_q_values: `torch.Tensor`, batch of next state Q-values.
            dones: `torch.Tensor`, batch indicating if the transition is
                terminal.
        
        Returns:
            `torch.Tensor`, target Q-values.
        """
        next_state_q_values = self._next_state_q_values(next_states)
        return rewards + self.gamma_n * next_state_q_values * (1 - dones)
    
    def learn(self):
        """Learns the Q-value from experience replay."""
        if len(self.replay_buffer) - self.n + 1 < self.batch_size:
            return
        
        self.replay_buffer.sample(self.batch_size)
        
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        dones = self.replay_buffer.dones
        
        state_q_values = self.network(states).gather(1, actions)
        target_state_q_values = self._target_state_q_values(rewards,
                                                            next_states,
                                                            dones)
        
        if self.per:
            errors = target_state_q_values - state_q_values
            for i, index in enumerate(self.replay_buffer.indices):
                self.replay_buffer.update(index, errors[i].item())
            
            loss = self.criterion(state_q_values, target_state_q_values)
            loss *= self.replay_buffer.is_weight
            loss = loss.mean()
        else:
            loss = self.criterion(state_q_values, target_state_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_network()
    
    def train(self, env, n_episodes):
        """Trains the agent in the environment for n_episodes episodes.
        
        Args:
            env: Gym environment.
            n_episodes: int, number of episodes to train for.
        
        Returns:
            list of floats, list of returns.
        """
        return_list = []
        for i_episode in range(1, n_episodes+1):
            episode_return = 0
            state = env.reset()
            for t in count():
                self._linearly_decay_epsilon()
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.replay_buffer.add(torch.tensor(state,
                                                    dtype=torch.float32),
                                       torch.tensor([action],
                                                    dtype=torch.long),
                                       reward,
                                       torch.tensor(next_state,
                                                    dtype=torch.float32),
                                       done)
                state = next_state
                episode_return += reward
                self.step += 1
                
                for _ in range(self.n_gradient_steps):
                    self.learn()
                
                if done:
                    return_list.append(episode_return)
                    print("Episode {:4d} : {:4d} steps | epsilon = {:4.2f} "
                          "| return = {:.1f}".format(i_episode, t+1,
                                                     self.epsilon,
                                                     episode_return))
                    
                    if return_list and episode_return >= max(return_list):
                        self.save("model.pt")
                    
                    break
        
        return return_list
    
    def test(self, env, n_steps=np.inf, agent_name=None):
        """Tests the agent in the environment for one episode and at most
        n_steps time steps.
        
        If agent_name is not None, record a video of the episode and write it
        to a disk file `videos/{env.spec.id}-{agent_name}.mp4`.
        
        Args:
            env: Gym environment.
            n_steps: int, maximum number of steps.
            agent_name: str, filename of the recorded video. If None, the
                episode is not recorded.
        """
        self.network.eval()
        if agent_name:
            if not os.path.exists("videos"):
                os.mkdir("videos")
            recorder = VideoRecorder(env,
                                     base_path="videos/{}-{}"
                                     .format(env.spec.id, agent_name))
            
        episode_return = 0
        state = env.reset()
        for t in count():
            if agent_name:
                recorder.capture_frame()
            else:
                env.render()
            
            action = self.greedy_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            state = next_state

            if done or t+1 >= n_steps:
                if agent_name:
                    recorder.close()
                env.close()
                self.network.train()
                return episode_return
    
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
            map_location: str, string specifying how to remap storage
                locations.
        """
        self.network.load_state_dict(torch.load(path,
                                                map_location=map_location))
