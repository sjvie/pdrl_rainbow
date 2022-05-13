import random
from collections import deque


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

def get_next_power_of_2(k):
    n = 2
    while n < k:
        n **= 2
    return n
