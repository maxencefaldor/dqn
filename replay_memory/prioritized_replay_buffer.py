import math
import numpy as np
from replay_memory.sum_tree import SumTree

class ReplayBuffer:
    def __init__(self,
                 buffer_size,
                 e=0.01, a=0.6,
                 beta=0.4,
                 beta_increment_per_sampling=0.001):
        
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.buffer_size = int(buffer_size)
        self.buffer = []
        self.index = 0
        self.sum_tree = SumTree(self.buffer_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
    
    def add(self, state, action, reward, next_state, done, error=None):
        if error is None:
            self.sum_tree.set(self.index, self.sum_tree.max_recorded_priority)
        else:
            priority = self._get_priority(error)
            self.sum_tree.set(self.index, priority)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.buffer_size
    
    def sample(self, batch_size):
        indices = self.sum_tree.stratified_sample(batch_size)
        priorities = [self.sum_tree.get(index) for index in indices]
        batch = [self.buffer[index] for index in indices]
        
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])
        
        sampling_probabilities = priorities / self.sum_tree.total()
        is_weight = np.power(self.buffer_size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        return indices, batch, is_weight
    
    def update(self, index, error):
        priority = self._get_priority(error)
        self.sum_tree.set(index, priority)
