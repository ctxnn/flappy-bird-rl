from collections import deque 
import random 

class ReplayMemory:
    def __init__(self, max_size, seed=None):
        self.memory = deque([], maxlen=max_size)
        
        if seed is not None:
            random.seed(seed)

    def appned(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()