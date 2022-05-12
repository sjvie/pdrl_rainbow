import random
from collections import deque


class ReplayBuffer():

    def __init__(self, size):
        self.size = size
        self.memory = deque([], maxlen=size)

    def add(self, state, action, reward, next_state, done):
        """ Add new experience to the buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def get_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def save(self, folder_name):
        # TODO
        pass

    def load(self, folder_name):
        # TODO
        pass

class PrioritzedBuffer(ReplayBuffer):
    # TODO: Nochmal schauen, welche Methoden wir hier noch brauchen
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        pass

    def get_batch(self, batch_size=32):
        """ sample batch from memory using a KL loss-based distribution """
        # TODO
        pass

    # TODO


