import random
import numpy as np


class PrioritizedBuffer:
    def __init__(self, size, observation_space, alpha, beta):
        """
        :param size (int): size of the replay buffer
        :param observation_space (int): size of the observations
        :param alpha (float): hyperparameter. Exponent of the priorities
        :param beta (float): hyperparameter. Exponent used in calculating the weights
        """

        self.alpha = alpha
        self.beta = beta
        self.max_size = size
        self.current_size = 0

        dt = np.dtype([("obs", np.uint8, (observation_space,)), ("action", np.uint8), ("reward", np.float32),
                       ("next_obs", np.uint8, (observation_space,)), ("done", bool)])

        self.memory = np.zeros(size, dt)

        self.tree = SumMinMaxTree(size)
        self.max_prio = 1.0
        self.idx = 0

    def add(self, state, action, reward, next_state, done):
        """ Add new experience to the buffer
        :param state (np array): state before the action. array with dim [observation_space]
        :param action (int): action that the agent chose
        :param reward (float): reward that the agent got from the environment
        :param next_state (np array): state after the action. array with dim [observation_space]
        :param done (bool): whether the experience ends the episode
        """

        # save experience to memory
        self.memory[self.idx] = (state, action, reward, next_state, done)

        # set the priority of the new experience
        if self.idx == 0:
            # when the buffer is empty just use value 1.0
            max_priority = 1.0
        else:
            # take the max priority of all experiences
            max_priority = self.tree.max()

        # add the experience to the tree
        self.tree.add(self.idx, max_priority ** self.alpha)

        # set new size and index
        if self.current_size < self.max_size:
            self.current_size += 1
        self.idx = (self.idx + 1) % self.max_size

    def get_batch(self, batch_size=32):
        """
        :param batch_size (int): size of the batch to be sampled
        :return ([experience], [float], [int]): list of experiences consisting of (state, action, reward, next_state, done)
                                                list of weights for training
                                                list of indices of the experiences in the buffer
        """

        states = [0] * batch_size
        actions = [0] * batch_size
        rewards = [0] * batch_size
        next_states = [0] * batch_size
        dones = [0] * batch_size
        weights = [0] * batch_size
        indices = [0] * batch_size

        # get total sum of priorities
        treesum = self.tree.sum()

        # the range from which each experience is sampled
        batch_range = treesum / batch_size

        # get the max weight using the minimum priority
        max_weight = ((1 / self.current_size) * (1 / self.tree.min())) ** self.beta

        for i in range(batch_size):
            # get the random priority to sample from the tree
            sample_priority = random.random() * batch_range + i * batch_range

            # sample from the tree
            idx, priority = self.tree.sample(sample_priority)

            # calculate the weight
            # w_j = (N* P(j))⁻beta  /max weight
            weight = ((self.current_size * priority) ** -self.beta) / max_weight

            weights[i] = weight
            indices[i] = idx
            states[i], actions[i], rewards[i], next_states[i], dones[i] = self.memory[idx]

        return (states, actions, rewards, next_states, dones), weights, indices

    def set_prio(self, idx, priority):
        """
        :param idx (int): index of the experience
        :param priority (float): new priority of the experience. Alpha hyperparameter is applied in this method
        """
        self.tree.set_priority(idx, priority ** self.alpha)

    def save(self, file_name):
        np.save(file_name + ".npy", self.memory)
        self.tree.save(file_name + ".npy")
        # todo
        pass

    def load(self, file_name):
        with open(file_name + ".npy", 'rb') as f:
            self.memory = np.load(f)
            self.tree.load(file_name + "_sum.npy")
        # todo
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
        self.min_array[current_index] = priority
        self.max_array[current_index] = priority
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
        while current_index < self.array_size - 1:
            left_index = current_index * 2 + 1
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
        n *= 2
    return n
