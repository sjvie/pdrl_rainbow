import math
import random
import numpy as np
import torch


class PrioritizedBuffer:
    def __init__(self, size, observation_shape, n_step_returns, alpha, beta, device, discount_factor,
                 tensor_memory=False):
        """
        :param size (int): size of the replay buffer
        :param observation_space (int): size of the observations
        :param alpha (float): hyperparameter. Exponent of the priorities
        :param beta (float): hyperparameter. Exponent used in calculating the weights
        """

        self.observation_shape = observation_shape
        self.n_step_returns = n_step_returns
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.discount_factor = discount_factor
        self.tensor_memory = tensor_memory
        self.max_size = size
        self.current_size = 0

        self.memory_size = size + n_step_returns
        if self.tensor_memory:
            self.obs_memory = torch.zeros((self.memory_size,) + observation_shape, dtype=torch.uint8,
                                          device=self.device)
            self.action_memory = torch.zeros(self.memory_size, dtype=torch.uint8, device=self.device)
            self.reward_memory = torch.zeros(self.memory_size, dtype=torch.float32, device=self.device)
            self.done_memory = torch.zeros(self.memory_size, dtype=torch.bool, device=self.device)
        else:
            dt = np.dtype(
                [("obs", np.uint8, observation_shape), ("action", np.uint8), ("reward", np.float32), ("done", bool)])
            self.memory = np.zeros(self.memory_size, dt)

        self.tree = SumMinMaxTree(size, tensor_memory=False, device=device)
        self.idx = 0
        self.tree_idx = 0

    def add(self, state, action, reward, done):
        """ Add new experience to the buffer
        # todo update doc
        :param state (np array): state before the action. array with dim [observation_space]
        :param action (int): action that the agent chose
        :param reward (float): reward that the agent got from the environment
        :param next_state (np array): state after the action. array with dim [observation_space]
        :param done (bool): whether the experience ends the episode
        """

        self.set_memory(self.idx, (state, action, reward, done))

        if self.current_size >= self.n_step_returns:
            # set the priority of the new experience
            if self.current_size == self.n_step_returns:
                # when the buffer is empty, just use value 1.0
                max_priority = 1.0
            else:
                # take the max priority of all experiences
                max_priority = self.tree.max()

            # add the experience to the tree
            self.tree.add(self.tree_idx, max_priority)
            self.tree_idx = (self.tree_idx + 1) % self.max_size

        # add to the rewards of the last n experiences
        for i in range(1, self.n_step_returns):
            r_idx = (self.idx - i) % self.memory_size
            self.add_reward(r_idx, reward * self.discount_factor ** i)

        # move index by one
        self.idx = (self.idx + 1) % self.memory_size

        # increase size until max_size is reached
        if self.current_size < self.memory_size:
            self.current_size += 1

    def get_batch(self, batch_size=32):
        """
        :param batch_size: (int): size of the batch to be sampled
        :return ([experience], [float], [int]): list of experiences consisting of (state, action, reward, n_next_state, done)
                                                list of weights for training
                                                list of indices of the experiences in the buffer
        """

        if self.tensor_memory:
            states = torch.empty((batch_size,) + self.observation_shape, dtype=torch.uint8, device=self.device)
            actions = torch.empty(batch_size, dtype=torch.uint8, device=self.device)
            rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
            n_next_states = torch.empty((batch_size,) + self.observation_shape, dtype=torch.uint8, device=self.device)
            dones = torch.empty(batch_size, dtype=torch.bool, device=self.device)
            weights = torch.empty(batch_size, dtype=torch.float32, device=self.device)
            indices = torch.empty(batch_size, dtype=torch.int32, device=self.device)
        else:
            states = np.empty((batch_size,) + self.observation_shape, dtype=np.uint8)
            actions = np.empty(batch_size, dtype=np.uint8)
            rewards = np.empty(batch_size, dtype=np.float32)
            n_next_states = np.empty((batch_size,) + self.observation_shape, dtype=np.uint8)
            dones = np.empty(batch_size, dtype=bool)
            weights = np.empty(batch_size, dtype=np.float32)
            indices = np.empty(batch_size, dtype=np.uint32)

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
            tree_idx, priority = self.tree.sample(sample_priority)

            # get the corresponding index for the memory
            idx = self.tree_idx_to_idx(tree_idx)

            # calculate the weight
            # w_j = (N* P(j))⁻beta  /max weight
            weight = ((self.current_size * priority) ** -self.beta) / max_weight

            weights[i] = weight
            indices[i] = idx
            states[i], actions[i], rewards[i], dones[i] = self.get_memory(idx)

            # get the state in n steps
            n_next_idx = (idx + self.n_step_returns) % self.memory_size
            n_next_states[i] = self.get_memory(n_next_idx, experience_part=0)

        return (states, actions, rewards, n_next_states, dones), weights, indices

    def idx_to_tree_idx(self, idx):
        return (self.tree_idx + (idx - self.idx) % self.memory_size) % self.max_size

    def tree_idx_to_idx(self, tree_idx):
        return (self.idx + (tree_idx - self.tree_idx) % self.max_size) % self.memory_size

    def add_reward(self, idx, reward):
        if self.tensor_memory:
            self.reward_memory[idx] += reward
        else:
            self.memory[idx][2] += reward

    def set_memory(self, idx, experience):
        if self.tensor_memory:
            self.obs_memory[idx], self.action_memory[idx], self.reward_memory[idx], self.done_memory[idx] = experience
        else:
            self.memory[idx] = experience

    def get_memory(self, idx, experience_part=None):
        if self.tensor_memory:
            if experience_part is None:
                return self.obs_memory[idx], self.action_memory[idx], self.reward_memory[idx], self.done_memory[idx]
            elif experience_part == 0:
                return self.obs_memory[idx]
            elif experience_part == 1:
                return self.action_memory[idx]
            elif experience_part == 2:
                return self.reward_memory[idx]
            elif experience_part == 3:
                return self.done_memory[idx]
        else:
            if experience_part is None:
                return self.memory[idx]
            else:
                return self.memory[idx][experience_part]

    def set_prio(self, idx, priority):
        """
        :param idx (int): index of the experience
        :param priority (float): new priority of the experience. Alpha hyperparameter is applied in this method
        """

        assert priority > 0
        self.tree.set_priority(self.idx_to_tree_idx(idx), priority ** self.alpha)

    def save(self, file_name):
        if self.tensor_memory:
            torch.save(self.obs_memory, file_name + "_obs.pt")
            torch.save(self.action_memory, file_name + "_action.pt")
            torch.save(self.reward_memory, file_name + "_reward.pt")
            torch.save(self.done_memory, file_name + "_done.pt")
        else:
            np.save(file_name + ".npy", self.memory)

        self.tree.save(file_name + "_tree")

    def load(self, file_name):
        if self.tensor_memory:
            self.obs_memory = torch.load(file_name + "_obs.pt").to(self.device)
            self.action_memory = torch.load(file_name + "_action.pt").to(self.device)
            self.reward_memory = torch.load(file_name + "_reward.pt").to(self.device)
            self.done_memory = torch.load(file_name + "_done.pt").to(self.device)
        else:
            self.memory = np.load(file_name + ".npy")

        self.tree.load(file_name + "_tree")


class SumMinMaxTree:

    def __init__(self, capacity, tensor_memory=False, device=None):
        self.capacity = capacity
        self.tensor_memory = tensor_memory
        self.device = device
        self.array_size = get_next_power_of_2(self.capacity)
        if tensor_memory:
            assert device is not None
            self.sum_array = torch.zeros(self.array_size * 2 - 1, dtype=torch.float32).to(device)
            self.min_array = torch.full((self.array_size * 2 - 1,), torch.inf, dtype=torch.float32).to(device)
            self.max_array = torch.zeros(self.array_size * 2 - 1, dtype=torch.float32).to(device)
        else:
            # todo: possible memory optimization (if needed) -> maybe float16
            self.sum_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)
            self.min_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
            self.min_array.fill(np.inf)
            self.max_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)

    def sum(self):
        return self.sum_array[0].item()

    def min(self):
        return self.min_array[0].item()

    def max(self):
        return self.max_array[0].item()

    def add(self, data_index, priority):
        self.set_priority(data_index, priority)

    def set_priority(self, data_index, priority):
        assert not math.isnan(priority)
        current_index = data_index + self.array_size - 1
        self.sum_array[current_index] = priority
        self.min_array[current_index] = priority
        self.max_array[current_index] = priority
        if self.tensor_memory:
            current_index = torch.div((current_index - 1), 2, rounding_mode='floor')
        else:
            current_index = (current_index - 1) // 2

        while current_index >= 0:
            child_index = current_index * 2 + 1
            self.sum_array[current_index] = self.sum_array[child_index] + self.sum_array[child_index + 1]
            self.min_array[current_index] = min(self.min_array[child_index], self.min_array[child_index + 1])
            self.max_array[current_index] = max(self.max_array[child_index], self.max_array[child_index + 1])

            if self.tensor_memory:
                current_index = torch.div((current_index - 1), 2, rounding_mode='floor')
            else:
                current_index = (current_index - 1) // 2

    def sample(self, sample_priority):
        assert 0 <= sample_priority < self.sum()

        current_index = 0
        while current_index < (self.array_size - 1):
            left_index = current_index * 2 + 1
            left_priority = self.sum_array[left_index]
            if left_priority >= sample_priority:
                current_index = left_index
            else:
                sample_priority -= left_priority
                current_index = left_index + 1

        data_index = current_index - (self.array_size - 1)
        assert 0 <= data_index < self.capacity
        return data_index, self.sum_array[current_index]

    def clear(self):
        if self.tensor_memory:
            assert self.device is not None
            self.sum_array = torch.zeros(self.array_size * 2 - 1, dtype=torch.float32).to(self.device)
            self.min_array = torch.full((self.array_size * 2 - 1,), torch.inf, dtype=torch.float32).to(self.device)
            self.max_array = torch.zeros(self.array_size * 2 - 1, dtype=torch.float32).to(self.device)
        else:
            # todo: possible memory optimization (if needed) -> maybe float16
            self.sum_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)
            self.min_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
            self.min_array.fill(np.inf)
            self.max_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)

    def save(self, file_name):
        if self.tensor_memory:
            torch.save(self.sum_array, file_name + "_sum.pt")
            torch.save(self.min_array, file_name + "_min.pt")
            torch.save(self.max_array, file_name + "_max.pt")
        else:
            np.save(file_name + "_sum.npy", self.sum_array)
            np.save(file_name + "_min.npy", self.min_array)
            np.save(file_name + "_max.npy", self.max_array)

    def load(self, file_name):
        if self.tensor_memory:
            self.sum_array = torch.load(file_name + "_sum.pt").to(self.device)
            self.min_array = torch.load(file_name + "_min.pt").to(self.device)
            self.max_array = torch.load(file_name + "_max.pt").to(self.device)
        else:
            self.sum_array = np.load(file_name + "_sum.npy")
            self.min_array = np.load(file_name + "_min.npy")
            self.max_array = np.load(file_name + "_max.npy")


def get_next_power_of_2(k):
    n = 2
    while n < k:
        n *= 2
    return n
