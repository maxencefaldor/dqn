#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The standard DQN replay memory.

The replay memory is a circular buffer, supporting multi-step bootstrapping.
"""

import math
import random
from collections import namedtuple
import torch


Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer(object):
    """Implementation of the standard DQN replay memory."""
    
    def __init__(self,
                 gamma,
                 n,
                 buffer_size=1e6):
        """Initializes the Replay Buffer.
        
        Args:
            gamma: float, discount rate.
            n: int, number of steps of bootstrapping.
            buffer_size: int, capacity of the buffer.
        """
        self._buffer_size = int(buffer_size)
        self._buffer = []
        self._index = 0
        self._n = n
        self._gamma = gamma
        self._cumulative_discount_vector = torch.tensor(
            [math.pow(self._gamma, n) for n in range(self._n)],
            dtype=torch.float32)
    
    def __len__(self):
        """Returns the length of the buffer.
        
        Returns:
            int, length of the buffer, less than _buffer_size
        """
        return len(self._buffer)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.
        
        Args:
            state: torch.Tensor, state of the agent at the beginning of the
            transition.
            action: torch.Tensor, action taken in state `state`.
            reward: float, reward received after taking action `action` in
                state `state`.
            next_state: torch.Tensor, new state of the agent after taking
            action `action` in state `state`.
            done: bool, equals True if the transition is terminal, else equals
                False.
        """
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(None)
        self._buffer[self._index] = (
            Transition(state, action, reward, next_state, done))
        self._index = (self._index + 1) % self._buffer_size
    
    def _sample_index(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.
        
        There are n - 1 invalid transitions before the cursor, because we do
        not have a valid n-step transition.
        
        Args:
            batch_size: int, batch size.
            
        Returns:
            list of ints, a batch of valid indices sampled uniformly.
        """
        if len(self._buffer) < self._buffer_size:
            return random.sample(
                range(0, self._index - self._n + 1),
                batch_size)
        else:
            shifted_indices = random.sample(
                range(self._index, self._index + self._buffer_size - self._n + 1),
                batch_size)
            return [index % self._buffer_size for index in shifted_indices]
    
    def sample(self, batch_size, indices=None):
        """Returns a batch of transitions.
        
        Args:
            batch_size: int, batch size.
            indices: None or list of ints, the indices of the transitions in
                the batch. If None, sample the indices uniformly.
        
        Returns:
            batch: namedtuple, batch of transitions.
        """
        batch = []
        if indices is None:
            indices = self._sample_index(batch_size)
        
        for index in indices:
            trajectory_indices = [(index + i) % self._buffer_size
                                  for i in range(self._n)]
            trajectory_dones = [self._buffer[i].done
                                for i in trajectory_indices]
            done = any(trajectory_dones)
            if done:
                trajectory_length = trajectory_dones.index(True) + 1
                trajectory_indices = trajectory_indices[:trajectory_length]
            else:
                trajectory_length = self._n
            
            trajectory_discount_vector = (
                self._cumulative_discount_vector[:trajectory_length])
            trajectory_rewards = torch.tensor(
                [self._buffer[i].reward for i in trajectory_indices])
            
            state = self._buffer[index].state
            action = self._buffer[index].action
            n_return = torch.sum(
                trajectory_discount_vector * trajectory_rewards, axis=0).unsqueeze(0)
            next_state = self._buffer[
                (index + trajectory_length - 1) % self._buffer_size].next_state
            done = torch.tensor([done], dtype=torch.float32)
            
            batch.append(Transition(state, action, n_return, next_state, done))
        return Transition(*zip(*batch))
