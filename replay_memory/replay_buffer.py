import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.buffer = []
        self.index = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.buffer_size
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
