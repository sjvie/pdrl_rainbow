import math
import random
import numpy as np
import torch


class Buffer:
    def __init__(self, size, observation_shape, conf):
        self.observation_shape = observation_shape
        self.n_step_returns = conf.multi_step_n
        self.device = conf.device
        self.discount_factor = conf.discount_factor
        self.max_size = size

        self.current_size = 0
        self.n_current_size = 0
        self.idx = 0
        self.n_idx = 0

        self.memory_size = size

        self.state_memory = np.zeros((self.memory_size,) + observation_shape, dtype=conf.obs_dtype)
        self.action_memory = np.zeros(self.memory_size, dtype=np.uint8)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.done_memory = np.zeros(self.memory_size, dtype=bool)
        self.n_next_state_memory = np.zeros((self.memory_size,) + observation_shape, dtype=conf.obs_dtype)

        self.n_memory_size = self.n_step_returns

        self.n_state_memory = np.zeros((self.n_memory_size,) + observation_shape, dtype=conf.obs_dtype)
        self.n_action_memory = np.zeros(self.n_memory_size, dtype=np.uint8)
        self.n_reward_memory = np.zeros(self.n_memory_size, dtype=np.float32)
        self.n_done_memory = np.zeros(self.n_memory_size, dtype=bool)

    def add(self, state, action, reward, done):
        self.state_memory[self.idx] = self.n_state_memory[self.n_idx]
        self.action_memory[self.idx] = self.n_action_memory[self.n_idx]
        self.reward_memory[self.idx] = self.n_reward_memory[self.n_idx]
        self.done_memory[self.idx] = self.n_done_memory[self.n_idx]
        self.n_next_state_memory[self.idx] = state

        self.n_state_memory[self.n_idx] = state
        self.n_action_memory[self.n_idx] = action
        self.n_reward_memory[self.n_idx] = reward
        self.n_done_memory[self.n_idx] = done

        # add to the rewards of the last n experiences
        for i in range(1, self.n_step_returns):
            r_idx = (self.n_idx - i) % self.n_memory_size

            self.n_reward_memory[r_idx] += reward * self.discount_factor ** i

        # move n idx by one
        self.n_idx = (self.n_idx + 1) % self.n_memory_size

        # increase size until max_size is reached
        if self.n_current_size < self.n_memory_size:
            self.n_current_size += 1
        else:
            # move idx by one
            self.idx = (self.idx + 1) % self.memory_size

            if self.current_size < self.memory_size:
                self.current_size += 1

    def get_batch(self, batch_size=32):
        """
        :param batch_size: (int): size of the batch to be sampled
        :return ([experience], [float], [int]): list of experiences consisting of (state, action, reward, n_next_state, done)
                                                list of weights for training
                                                list of indices of the experiences in the buffer
        """
        assert self.current_size >= batch_size

        indices = torch.randint(low=0, high=self.current_size, size=(batch_size,), device=self.device)
        np_indices = indices.cpu().numpy()

        # states, actions, rewards, dones, n_next_states = self.get_experiences(indices)
        np_states, np_actions, np_rewards, np_dones, np_n_next_states = self.get_experiences(np_indices)

        return (np_states, np_actions, np_rewards, np_n_next_states, np_dones), None, np_indices


    def get_experiences(self, idxs):
        return self.state_memory[idxs], self.action_memory[idxs], self.reward_memory[idxs], \
               self.done_memory[idxs], self.n_next_state_memory[idxs]

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

    def set_prio(self, param, param1):
        raise NotImplementedError("this method is not implemented, consider using PrioritizedBuffer")


class PrioritizedBuffer(Buffer):

    def __init__(self, size, observation_shape, conf):
        super().__init__(size, observation_shape, conf)

        self.alpha = conf.replay_buffer_alpha
        self.beta = conf.replay_buffer_beta_start
        self.beta_end = conf.replay_buffer_beta_end
        self.beta_annealing_steps = conf.replay_buffer_beta_annealing_steps
        self.delta_beta = (self.beta_end - self.beta) / self.beta_annealing_steps

        self.initial_max_priority = conf.per_initial_max_priority

        self.tree = SumMinMaxTree(self.memory_size)
        self.max_priority = self.initial_max_priority

    def add(self, state, action, reward, done):
        if self.n_current_size >= self.n_memory_size:
            # add the nth newest experience to the tree
            self.tree.add(self.idx, self.max_priority ** self.alpha)

        super().add(state, action, reward, done)

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

        indices = np.empty(batch_size, dtype=np.int32)

        # get total sum of priorities
        treesum = self.tree.sum()

        # the range from which each experience is sampled
        batch_range = treesum / batch_size

        # get the max weight using the minimum priority
        # todo: some other implementations just use the max weight of the batch
        #       what does this change?

        min_probability = self.tree.min() / self.tree.sum()
        max_weight = (self.current_size * min_probability) ** (-self.beta)

        priorities = np.empty(batch_size, dtype=np.float32)

        for i in range(batch_size):
            # get the random priority to sample from the tree
            sample_priority = random.random() * batch_range + i * batch_range

            # sample from the tree
            indices[i], priorities[i] = self.tree.sample(sample_priority)

            assert 0 <= indices[i] < self.current_size, indices

        # get experiences from indices
        states, actions, rewards, dones, n_next_states = self.get_experiences(indices)

        # get probabilities from priorities
        weights = priorities / self.tree.sum()

        # calculate weights from probabilities
        # w_j = (N* P(j))⁻beta  /max weight
        weights = ((self.current_size * weights) ** (-self.beta)) / max_weight

        return (states, actions, rewards, n_next_states, dones), weights, indices

    def set_prio(self, idx, priority):
        assert not math.isnan(priority)
        assert priority > 0
        self.max_priority = max(self.max_priority, priority)
        self.tree.set_priority(idx, priority ** self.alpha)

    def save(self, file_name):
        super().save(file_name)
        self.tree.save(file_name + "_tree")

    def load(self, file_name):
        super().load(file_name)
        self.tree.load(file_name + "_tree")


class SumMinMaxTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.current_size = 0
        self.array_size = get_next_power_of_2(self.capacity)
        self.data_index_offset = self.array_size - 1
        self.sum_array = np.zeros(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array = np.empty(self.array_size * 2 - 1, dtype=np.float32)
        self.min_array.fill(np.inf)

    def sum(self):
        return self.sum_array[0].item()

    def min(self):
        return self.min_array[0].item()

    def add(self, data_index, priority):
        self.set_priority(data_index, priority)
        if self.current_size < self.capacity:
            self.current_size += 1

    def get_priority(self, idx):
        return self.sum_array[self.data_index_offset + idx]

    def _set_priority(self, data_index, sum_priority, min_priority):
        current_index = data_index + self.data_index_offset
        self.sum_array[current_index] = sum_priority
        self.min_array[current_index] = min_priority
        current_index = (current_index - 1) // 2

        while current_index >= 0:
            child_index = current_index * 2 + 1
            self.sum_array[current_index] = self.sum_array[child_index] + self.sum_array[child_index + 1]
            self.min_array[current_index] = min(self.min_array[child_index], self.min_array[child_index + 1])

            current_index = (current_index - 1) // 2

    def set_priority(self, data_index, priority):
        assert not math.isnan(priority)
        self._set_priority(data_index, priority, priority)

    def reset_priority(self, data_index):
        self._set_priority(data_index, 0, np.inf)

    def sample(self, sample_priority):
        # add 1e-6 to account for floating point errors
        assert 0 <= sample_priority <= self.sum() + 1e-6
        assert self.current_size > 0

        current_index = 0
        while current_index < self.data_index_offset:
            left_index = current_index * 2 + 1
            left_priority = self.sum_array[left_index]
            if left_priority >= sample_priority:
                current_index = left_index
            else:
                sample_priority -= left_priority
                current_index = left_index + 1

        # clamp index
        current_index = min(current_index, self.current_size - 1 + self.data_index_offset)
        current_index = max(current_index, 0)

        data_index = current_index - self.data_index_offset
        return data_index, self.sum_array[current_index]

    def save(self, file_name):
        np.save(file_name + "_sum.npy", self.sum_array)
        np.save(file_name + "_min.npy", self.min_array)

    def load(self, file_name):
        self.sum_array = np.load(file_name + "_sum.npy")
        self.min_array = np.load(file_name + "_min.npy")


def get_next_power_of_2(k):
    n = 1
    while n < k:
        n *= 2
    return n
