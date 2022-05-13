import random
from collections import deque

import numpy as np


class PrioritizedBuffer:
    """
    :param obs: (Must be cast to numpy int!) observation which is "observed" by the agent, most likely 2D screen, In case of Atari 84x84 (x4 Frame Stack)
    :param next_obs: (Must be cast to numpy int!) the observation which resulted from action take in observation
    :param action:(discrete type, numpy int8) action taken in observation
    :param done: (type: numpy bool) parameter which indicates if the episode ended (0) or is ongoing (1)
    :param weight: (numpy float) importance weights for every element in batch, which indicates the importance of the element.
        gradient is also multiplied with importance weight.
    :param alpha: weights the importance of every experience
    :param beta: adjusts weights
    """

    def __init__(self, size, observation_space, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.max_size = size
        self.current_size = 0

        dt = np.dtype([("obs", np.uint8, (observation_space,)), ("action", np.uint8), ("reward", np.float32),
                       ("next_obs", np.uint8, (observation_space,)), ("done", np.bool)])

        self.memory = np.zeros(size, dt)

        self.sumtree = SumMinMaxTree(size)
        self.max_prio = 1.0
        self.idx = 0

    def add(self, state, action, reward, next_state, done):
        """ Add new experience to the buffer"""
        self.memory[self.idx] = (state, action, reward, next_state, done)

        max_priority = self.sumtree.max()
        self.sumtree.add(self.idx, max_priority ** self.alpha)
        if self.current_size < self.max_size:
            self.current_size += 1
        self.idx = (self.idx + 1) % self.max_size

    def get_batch(self, batch_size=32):
        batch = []
        weights = []
        idxs = []
        treesum = self.sumtree.sum()
        batch_range = treesum / batch_size

        max_weight = ((1 / self.current_size) * (1 / self.sumtree.min())) ** self.beta

        for i in range(batch_size):
            sample_priority = random.random() * batch_range + i * batch_range
            idx, priority = self.sumtree.sample(sample_priority)
            # w_j = (N* P(j))⁻beta  /max weight
            weight = ((self.current_size * priority) ** -self.beta) / max_weight
            weights.append(weight)
            idxs.append(idx)
            batch.append(self.memory[idx])

        return batch, weights, idxs

    def set_prio(self, idx, priority):
        self.sumtree.set_priority(idx, priority ** self.alpha)

    def save(self, file_name):
        np.save(file_name + ".npy", self.memory)
        self.sumtree.save(file_name + ".npy")
        pass

    def load(self, file_name):
        with open(file_name + ".npy", 'rb') as f:
            self.memory = np.load(f)
            self.sumtree.load(file_name + "_sum.npy")
        pass


class SumMinMaxTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.array_size = get_next_power_of_2(self.capacity)
        # todo: possible memory optimization (if needed) -> maybe float16
        self.sum_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array.fill(np.inf)
        self.max_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)

    def sum(self):
        return self.sum_array[0]

    def min(self):
        return self.min_array[0]

    def max(self):
        return self.max_array[0]

    def add(self, data_index, priority):
        self.set_priority(data_index, priority)

    def set_priority(self, data_index, priority):
        current_index = data_index + self.array_size - 1
        priority_diff = priority - self.sum_array[current_index]
        self.sum_array[current_index] = priority
        current_index = (current_index - 1) // 2

        while current_index >= 0:
            self.sum_array[current_index] += priority_diff
            child_index = current_index * 2 + 1
            self.min_array[current_index] = min(self.min_array[child_index], self.min_array[child_index + 1])
            self.max_array[current_index] = max(self.max_array[child_index], self.max_array[child_index + 1])

            current_index = (current_index - 1) // 2

    def sample(self, sample_priority):
        # todo: <= or <?
        assert 0 <= sample_priority < self.sum()

        current_index = 0
        left_index = current_index * 2 + 1
        while current_index < self.array_size:
            left_priority = self.sum_array[left_index]
            if left_priority > sample_priority:
                current_index = left_index
            else:
                sample_priority -= left_priority
                current_index = left_index + 1

        data_index = current_index - self.array_size - 1
        return data_index, self.sum_array[current_index]

    def clear(self):
        self.sum_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array.fill(np.inf)
        self.max_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)


def get_next_power_of_2(k):
    n = 2
    while n < k:
        n **= 2
    return n
