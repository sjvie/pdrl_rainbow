import math
import random
import numpy as np


class PrioritizedBuffer:
    def __init__(self, size, observation_space, n_step_returns, alpha, beta, observation_dt=np.uint8):
        """
        :param size (int): size of the replay buffer
        :param observation_space (int): size of the observations
        :param alpha (float): hyperparameter. Exponent of the priorities
        :param beta (float): hyperparameter. Exponent used in calculating the weights
        """

        self.observation_space = observation_space
        self.n_step_returns = n_step_returns
        self.alpha = alpha
        self.beta = beta
        self.max_size = size
        self.current_size = 0

        dt = np.dtype(
            [("obs", observation_dt, (observation_space,)), ("action", np.uint8), ("reward", np.float32), ("done", bool)])

        self.memory = np.zeros(size, dt)
        self.n_memory = np.zeros(n_step_returns - 1, dt)

        self.tree = SumMinMaxTree(size)
        self.idx = 0
        self.n_idx = 0
        self.n_size = 0
        self.n_max_size = n_step_returns - 1

    def add(self, state, action, reward, done):
        """ Add new experience to the buffer
        :param state (np array): state before the action. array with dim [observation_space]
        :param action (int): action that the agent chose
        :param reward (float): reward that the agent got from the environment
        :param next_state (np array): state after the action. array with dim [observation_space]
        :param done (bool): whether the experience ends the episode
        """

        if self.n_size >= self.n_step_returns - 1:

            # move oldest experience from n_memory to memory
            self.memory[self.idx] = self.n_memory[self.n_idx]

            # set the priority of the new experience
            if self.current_size == 0:
                # when the buffer is empty just use value 1.0
                max_priority = 1.0
            else:
                # take the max priority of all experiences
                max_priority = self.tree.max()

            # add the experience to the tree
            self.tree.add(self.idx, max_priority ** self.alpha)

            # move index by one
            self.idx = (self.idx + 1) % self.max_size

            # increase size until max_size is reached
            if self.current_size < self.max_size:
                self.current_size += 1

        else:
            self.n_size += 1

        # save new experience to n_memory
        self.n_memory[self.n_idx] = (state, action, reward, done)

        # move n_index by one
        self.n_idx = (self.n_idx + 1) % self.n_max_size

    def get_batch(self, batch_size=32):
        """
        :param batch_size: (int): size of the batch to be sampled
        :return ([experience], [float], [int]): list of experiences consisting of (state, action, reward, n_next_state, done)
                                                list of weights for training
                                                list of indices of the experiences in the buffer
        """

        states = np.zeros((batch_size, self.observation_space), dtype=np.uint8)
        actions = np.zeros(batch_size, dtype=np.uint8)
        rewards = np.zeros((batch_size, self.n_step_returns), dtype=np.float32)
        n_next_states = np.zeros((batch_size, self.observation_space), dtype=np.uint8)
        dones = np.zeros(batch_size, bool)
        weights = np.zeros(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.uint8)

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
            states[i], actions[i], rewards[i, 0], dones[i] = self.memory[idx]

            # n step returns
            # first, take the next (n-1) rewards from the main memory
            n = 1
            while n < self.n_step_returns:
                mem_idx = (idx + n) % self.max_size

                # if the end of the memory is reached, stop
                if mem_idx == self.idx:
                    break

                # take the state after n steps
                if n == self.n_step_returns:
                    n_next_states[i] = self.memory[mem_idx][0]

                rewards[i, n] = self.memory[mem_idx][2]
                n += 1

            # if the previous loop did not finish, take the next rewards from the n_memory
            n_mem_idx_offset = n
            while n < self.n_step_returns:
                n_mem_idx = (n - n_mem_idx_offset) % self.n_max_size

                # take the state after n steps
                if n == self.n_step_returns:
                    n_next_states[i] = self.memory[mem_idx][0]

                rewards[i, n] = self.n_memory[n_mem_idx][2]
                n += 1

        return (states, actions, rewards, n_next_states, dones), weights, indices

    def set_prio(self, idx, priority):
        """
        :param idx (int): index of the experience
        :param priority (float): new priority of the experience. Alpha hyperparameter is applied in this method
        """

        assert priority > 0
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
        assert not math.isnan(priority)
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
        assert 0 <= sample_priority < self.sum()

        current_index = 0
        while current_index < (self.array_size - 1):
            left_index = current_index * 2 + 1
            left_priority = self.sum_array[left_index]
            if left_priority > sample_priority:
                current_index = left_index
            else:
                sample_priority -= left_priority
                current_index = left_index + 1

        data_index = current_index - (self.array_size - 1)
        assert 0 <= data_index < self.capacity
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
