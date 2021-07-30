#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prioritized replay memory.

This implementation is based on the paper "Prioritized Experience Replay" by
Tom Schaul et al. (2015).
"""

import numpy as np
from replay_memory.replay_buffer import ReplayBuffer
from replay_memory.sum_tree import SumTree


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of the prioritized replay memory."""
    
    def __init__(self,
                 gamma,
                 n,
                 buffer_size=1e6,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment_per_sampling=0.001,
                 epsilon=0.01):
        """Initializes the Prioritized Replay Buffer.
        
        Args:
            gamma: float, discount rate.
            n: int, number of steps of bootstrapping.
            buffer_size: int, capacity of the buffer.
            alpha: float, prioritization exponent.
            beta: float, prioritization importance sampling.
            beta_increment_per_sampling: float, beta increment per sampling.
            epsilon: float, small number ensuring priorities are strictly
                greater than 0.
        """
        ReplayBuffer.__init__(self, gamma, n, buffer_size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.sum_tree = SumTree(self._buffer_size)
    
    def _get_priority(self, error):
        """Returns the transition priority.
        
        Args:
            error: float, TD-error.
        """
        return (np.abs(error) + self.epsilon) ** self.alpha
    
    def add(self, state, action, reward, next_state, done, error=None):
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
            error: None or float, TD-error. If None, set the maximum recorded
                priority as described in Schaul et al. (2015).
        """
        if error is None:
            self.sum_tree.set(self._index, self.sum_tree.max_recorded_priority)
        else:
            self.update(self._index, error)
        
        ReplayBuffer.add(self, state, action, reward, next_state, done)
    
    def update(self, index, error):
        """Update the priority of a transition.
        
        Args:
            index: int, index of the transition.
            error: float, TD-error.
        """
        priority = self._get_priority(error)
        self.sum_tree.set(index, priority)
    
    def _sample_index(self, batch_size):
        """Returns a batch of valid indices sampled as in Schaul et al. (2015).
        
        There are n - 1 invalid transitions before the cursor, because we do
        not have a valid n-step transition.
        
        Args:
            batch_size: int, batch size.
            
        Returns:
            list of ints, a batch of valid indices sampled with prioritization.
        """
        indices = self.sum_tree.stratified_sample(batch_size)
        invalid_indices = [i % self._buffer_size for i in range(self._index - self._n + 1, self._index)]
        
        for i in range(len(indices)):
            if indices[i] in invalid_indices:
                index = indices[i]
                while index in invalid_indices:
                    index = self.sum_tree.sample()
                indices[i] = index
        return indices
    
    def sample(self, batch_size):
        """Returns a batch of transitions, the indices and importance sampling
        weight of the transitions.
        
        Args:
            batch_size: int, batch size.
        
        Returns:
            indices: list of ints, list of indices of the transitions.
            batch: namedtuple, batch of transitions.
            is_weight: list of floats, importance sampling weights of the
                transitions.
        """
        indices = self._sample_index(batch_size)
        batch = ReplayBuffer.sample(self, batch_size, indices)
        priorities = [self.sum_tree.get(index) for index in indices]
        
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])
        
        sampling_probabilities = priorities / self.sum_tree.total()
        is_weight = np.power(self._buffer_size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        return indices, batch, is_weight
