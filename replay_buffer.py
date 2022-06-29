import math
import random
import numpy as np
import torch


class Buffer:
    def __init__(self, size, observation_shape, device, conf):
        self.observation_shape = observation_shape
        self.n_step_returns = conf.multi_step_n
        self.device = device
        self.discount_factor = conf.discount_factor
        self.max_size = size

        self.current_size = 0
        self.idx = 0

        self.memory_size = size + self.n_step_returns

        self.state_memory = torch.zeros((self.memory_size,) + observation_shape, dtype=torch.uint8,
                                        device=self.device)
        self.action_memory = torch.zeros(self.memory_size, dtype=torch.uint8, device=self.device)
        self.reward_memory = torch.zeros(self.memory_size, dtype=torch.float32, device=self.device)
        self.done_memory = torch.zeros(self.memory_size, dtype=torch.bool, device=self.device)

    def add(self, state, action, reward, done):
        self.state_memory[self.idx] = state
        self.action_memory[self.idx] = action
        self.reward_memory[self.idx] = reward
        self.done_memory[self.idx] = done

        # add to the rewards of the last n experiences
        for i in range(1, self.n_step_returns):
            r_idx = (self.idx - i) % self.memory_size

            self.reward_memory[r_idx] += reward * self.discount_factor ** i

        # move index by one
        self.idx = (self.idx + 1) % self.memory_size

        # increase size until max_size is reached
        if self.current_size < self.max_size:
            self.current_size += 1

    def get_batch(self, batch_size=32):
        """
        :param batch_size: (int): size of the batch to be sampled
        :return ([experience], [float], [int]): list of experiences consisting of (state, action, reward, n_next_state, done)
                                                list of weights for training
                                                list of indices of the experiences in the buffer
        """

        states, actions, rewards, n_next_states, dones, weights, indices = self.init_empty_batch(batch_size)

        for i in range(batch_size):
            idx = (self.idx - random.randint(1, self.current_size)) % self.memory_size
            weights[i] = 1

            states[i], actions[i], rewards[i], dones[i], n_next_states[i] = self.get_experience(idx)
            indices[i] = idx

        return (states, actions, rewards, n_next_states, dones), weights, indices

    def get_experience(self, idx):
        # get index of the state in n steps
        n_next_idx = (idx + self.n_step_returns) % self.memory_size
        return self.state_memory[idx], self.action_memory[idx], self.reward_memory[idx], self.done_memory[idx], \
               self.state_memory[n_next_idx]

    def init_empty_batch(self, batch_size):
        states = torch.empty((batch_size,) + self.observation_shape, dtype=torch.uint8, device=self.device)
        actions = torch.empty(batch_size, dtype=torch.uint8, device=self.device)
        rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        n_next_states = torch.empty((batch_size,) + self.observation_shape, dtype=torch.uint8, device=self.device)
        dones = torch.empty(batch_size, dtype=torch.bool, device=self.device)
        weights = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        indices = torch.empty(batch_size, dtype=torch.int32, device=self.device)

        return states, actions, rewards, n_next_states, dones, weights, indices

    def save(self, file_name):
        torch.save(self.state_memory, file_name + "_obs.pt")
        torch.save(self.action_memory, file_name + "_action.pt")
        torch.save(self.reward_memory, file_name + "_reward.pt")
        torch.save(self.done_memory, file_name + "_done.pt")

    def load(self, file_name):
        self.state_memory = torch.load(file_name + "_obs.pt").to(self.device)
        self.action_memory = torch.load(file_name + "_action.pt").to(self.device)
        self.reward_memory = torch.load(file_name + "_reward.pt").to(self.device)
        self.done_memory = torch.load(file_name + "_done.pt").to(self.device)


class PrioritizedBuffer(Buffer):

    def __init__(self, size, observation_shape, device, conf):
        super().__init__(size, observation_shape, device, conf)

        self.alpha = conf.replay_buffer_alpha
        self.beta = conf.replay_buffer_beta_start
        self.beta_end = conf.replay_buffer_beta_end
        self.beta_annealing_steps = conf.replay_buffer_beta_annealing_steps
        self.delta_beta = (self.beta_end - self.beta) / self.beta_annealing_steps

        self.initial_max_priority = conf.per_initial_max_priority

        self.tree = SumMinMaxTree(size)
        self.tree_idx = 0

    def add(self, state, action, reward, done):
        super().add(state, action, reward, done)

        if self.current_size > self.n_step_returns:
            # set the priority of the new experience
            # take the max priority of all experiences
            max_priority = self.tree.max()
            if max_priority == -np.inf:
                # when the buffer is empty, just use value 1.0
                max_priority = self.initial_max_priority

            # add the experience to the tree
            self.tree.add(self.tree_idx, max_priority)
            self.tree_idx = (self.tree_idx + 1) % self.max_size

        self.beta += self.delta_beta
        if self.beta > self.beta_end:
            self.beta = self.beta_end

    def get_batch(self, batch_size=32):
        """
        :param batch_size: (int): size of the batch to be sampled
        :return ([experience], [float], [int]): list of experiences consisting of (state, action, reward, n_next_state, done)
                                                list of weights for training
                                                list of indices of the experiences in the buffer
        """

        states, actions, rewards, n_next_states, dones, weights, indices = self.init_empty_batch(batch_size)

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
            weights[i] = ((self.current_size * priority) ** -self.beta) / max_weight

            states[i], actions[i], rewards[i], dones[i], n_next_states[i] = self.get_experience(idx)
            indices[i] = idx

        return (states, actions, rewards, n_next_states, dones), weights, indices

    def idx_to_tree_idx(self, idx):
        return (self.tree_idx + (idx - self.idx) % self.memory_size) % self.max_size

    def tree_idx_to_idx(self, tree_idx):
        return (self.idx + (tree_idx - self.tree_idx) % self.max_size) % self.memory_size

    def set_prio(self, idx, priority):
        """
        :param idx (int): index of the experience
        :param priority (float): new priority of the experience. Alpha hyperparameter is applied in this method
        """

        assert priority > 0
        self.tree.set_priority(self.idx_to_tree_idx(idx), priority ** self.alpha)

    def save(self, file_name):
        super().save(file_name)
        self.tree.save(file_name + "_tree")

    def load(self, file_name):
        super().load(file_name)
        self.tree.load(file_name + "_tree")


class SumMinMaxTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.array_size = get_next_power_of_2(self.capacity)
        self.data_index_offset = self.array_size - 1
        # todo: possible memory optimization (if needed) -> maybe float16
        self.sum_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array.fill(np.inf)
        self.max_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
        self.max_array.fill(-np.inf)

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
        current_index = data_index + self.data_index_offset
        self.sum_array[current_index] = priority
        self.min_array[current_index] = priority
        self.max_array[current_index] = priority
        current_index = (current_index - 1) // 2

        while current_index >= 0:
            child_index = current_index * 2 + 1
            self.sum_array[current_index] = self.sum_array[child_index] + self.sum_array[child_index + 1]
            self.min_array[current_index] = min(self.min_array[child_index], self.min_array[child_index + 1])
            self.max_array[current_index] = max(self.max_array[child_index], self.max_array[child_index + 1])

            current_index = (current_index - 1) // 2

    def sample(self, sample_priority):
        assert 0 <= sample_priority <= self.sum()

        current_index = 0
        while current_index < self.data_index_offset:
            left_index = current_index * 2 + 1
            left_priority = self.sum_array[left_index]
            if left_priority >= sample_priority:
                current_index = left_index
            else:
                sample_priority -= left_priority
                current_index = left_index + 1

        data_index = current_index - self.data_index_offset
        return data_index, self.sum_array[current_index]

    def save(self, file_name):
        np.save(file_name + "_sum.npy", self.sum_array)
        np.save(file_name + "_min.npy", self.min_array)
        np.save(file_name + "_max.npy", self.max_array)

    def load(self, file_name):
        self.sum_array = np.load(file_name + "_sum.npy")
        self.min_array = np.load(file_name + "_min.npy")
        self.max_array = np.load(file_name + "_max.npy")


def get_next_power_of_2(k):
    n = 2
    while n < k:
        n *= 2
    return n
