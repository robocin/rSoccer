import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class AverageBuffer:
    def __init__(self, capacity = 100):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
    
    def push(self, goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
         
        self.buffer[self.index] = goal
        self.index = (self.index + 1) % self.capacity
    
    def average(self):
        return np.mean(self.buffer if len(self.buffer) > 0 else [0])
    
    def __len__(self):
        return len(self.buffer)