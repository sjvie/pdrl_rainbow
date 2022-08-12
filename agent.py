import functools

from torch import nn
import torch
import numpy as np
import loss_functions
from reward_model import RND

from model import RainbowModel, NoisyLinear, ImpalaModel, D2RLModel, D2RLImpalaModel
from replay_buffer import PrioritizedBuffer, Buffer


class Agent:

    def __init__(self, observation_shape, action_space, conf):
        self.action_space = action_space
        self.in_channels = conf.frame_stack

        self.batch_size = conf.batch_size
        self.device = conf.device
        self.use_exploration = conf.use_exploration
        self.use_distributional = conf.use_distributional
        self.num_atoms = conf.distributional_atoms
        self.v_min = conf.distributional_v_min
        self.v_max = conf.distributional_v_max
        self.z_delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_support = torch.arange(self.v_min, self.v_max + self.z_delta / 2, self.z_delta, device=self.device)
        self.index_offset = torch.arange(0, self.batch_size, 1 / self.num_atoms,
                                         device=self.device).long() * self.num_atoms

        if self.use_distributional:
            self.get_loss = loss_functions.get_distributional_loss
        else:
            self.get_loss = loss_functions.get_huber_loss
            self.num_atoms = 1
            self.z_support = torch.tensor([1], device=self.device)

        self.n_step_returns = conf.multi_step_n
        self.discount_factor = conf.discount_factor
        self.use_rnd = conf.use_rnd
        self.use_noisy = conf.use_noisy
        self.epsilon = conf.epsilon_start

        self.exp_beta = conf.exp_beta_start
        self.exp_beta_mid = conf.exp_beta_mid
        self.exp_beta_end = conf.exp_beta_end
        self.exp_beta_annealing_steps = conf.exp_beta_annealing_steps
        self.exp_beta_annealing_steps2 = conf.exp_beta_annealing_steps
        self.delta_exp_beta = (self.exp_beta_mid - self.exp_beta) / self.exp_beta_annealing_steps
        self.delta_exp_beta2 = (self.exp_beta_end - self.exp_beta_mid) / self.exp_beta_annealing_steps

        self.adam_learning_rate = conf.adam_learning_rate
        self.adam_e = conf.adam_e

        self.use_per = conf.use_per
        self.replay_buffer_beta = conf.replay_buffer_beta_start
        self.replay_buffer_alpha = conf.replay_buffer_alpha
        self.replay_buffer_size = conf.replay_buffer_size
        self.replay_buffer_prio_offset = conf.replay_buffer_prio_offset
        self.num_envs = conf.num_envs
        self.num_envs_indexes = [i for i in range(self.num_envs)]
        self.use_kl_loss = conf.use_kl_loss
        self.grad_clip = conf.grad_clip
        if self.use_rnd:
            self.reward_model = RND(conf,
                                    action_space,
                                    nn.Linear,
                                    self.in_channels
                                    )

        if self.use_per:
            self.replay_buffer = PrioritizedBuffer(self.replay_buffer_size,
                                                   observation_shape,
                                                   conf
                                                   )
        else:
            self.replay_buffer = Buffer(self.replay_buffer_size,
                                        observation_shape,
                                        conf
                                        )

        self.use_double = conf.use_double

        if self.use_noisy:
            linear_layer = functools.partial(NoisyLinear, sigma_zero=conf.noisy_sigma_zero)
        else:
            linear_layer = nn.Linear

        if conf.model_arch == "rainbow":
            model_cls = RainbowModel
        elif conf.model_arch == "impala":
            model_cls = ImpalaModel
        elif conf.model_arch == "d2rl":
            model_cls = D2RLModel
        elif conf.model_arch == "d2rl_impala":
            model_cls = D2RLImpalaModel
        else:
            raise ValueError

        self.model = model_cls(conf=conf,
                               action_space=self.action_space,
                               linear_layer=linear_layer,
                               in_channels=self.in_channels
                               )

        if self.use_double:
            self.target_model = model_cls(conf=conf,
                                          action_space=self.action_space,
                                          linear_layer=linear_layer,
                                          in_channels=self.in_channels
                                          )
            self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.adam_learning_rate,
                                          eps=self.adam_e)

    def update_target_model(self):
        assert self.use_double
        self.target_model.load_state_dict(self.model.state_dict())

    def add_transitions(self, states, actions, rewards, dones):
        self.replay_buffer.add(states, actions, rewards, dones)

    def train(self):
        batch, weights, idxs = self.replay_buffer.get_batch(batch_size=self.batch_size)
        states, actions, rewards, n_next_states, dones = batch

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        n_next_states = torch.from_numpy(n_next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        idxs = torch.from_numpy(idxs).to(self.device)

        actions = actions.long()
        dones = dones.long()
        idxs = idxs.long()
        reward_loss = []
        if states.dtype == torch.uint8:
            states = states / 255.0
            n_next_states = n_next_states / 255.0

        self.model.generate_noise()
        self.target_model.generate_noise()

        if self.use_rnd:
            reward_loss = self.reward_model.train(n_next_states)

        loss, priorities = self.get_loss(self, states, actions, rewards, n_next_states, dones)

        loss = loss.squeeze()
        loss_copy = loss.clone()

        # update the priorities in the replay buffer
        if self.use_per:
            weights = torch.from_numpy(weights).to(self.device)
            for i in range(self.batch_size):
                # in the PER paper they used a small constant to prevent that the loss is 0
                self.replay_buffer.set_prio(idxs[i].item(), priorities[i].item())
            loss = loss * weights

        # use the average loss of the batch
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss_copy, weights, reward_loss

    def select_action(self, np_states, action_prob):
        """
        :param action_prob: tensor with probability distribution (in our case uniform distribution)
        :param np_states: (np array) numpy array with shape observation_shape
        :return (int): index of the selected action
        """



        states = torch.from_numpy(np_states).to(self.device)
        if states.dtype == torch.uint8:
            states = states / 255.0

        self.model.generate_noise()

        if not self.use_exploration:
            with torch.no_grad():
                if self.use_distributional:
                    q_dist = self.model(states)
                    q = (q_dist * self.z_support).sum(dim=-1)
                else:
                    q = self.model(states)

                actions = torch.argmax(q, dim=-1).cpu().numpy()

            # epsilon greedy
            if not self.use_noisy:
                use_random_actions = np.random.rand(actions.shape[0]) < self.epsilon
                random_actions = np.random.randint(0, self.action_space, actions.shape[0])
                actions[use_random_actions] = random_actions[use_random_actions]
            # return the indices of the actions
            return actions

        else:
            # todo: update for multiple states and actions
            #       and return np actions
            #action_prob = action_prob.expand(-1,self.num_envs)
            action_q_values = self.model(states, log=False)
            # since we use the softmax for these values, we can "normalize" them by subtracting the maximum
            # from each value as it still preserves the order of magnitude
            # this also prevents possible overflows (since the softmax function uses the e-function)
            max_action_prob = torch.max(action_q_values,dim=-1)
            max_action_prob = max_action_prob[0]
            max_action_prob = max_action_prob[:,None]
            max_action_prob = max_action_prob.expand(-1,self.action_space)
            action_probabilities = torch.add(-1 * max_action_prob,action_q_values)

            # we can clip the lower bound for action values, since they (most-likely) are not considered anyway
            # also used in https://openreview.net/pdf?id=HyEtjoCqFX
            action_probabilities = torch.clip(action_probabilities, min=-10, max=None)

            # we sample our actions after the softmax policy :
            # pi*(a|s) = exp(exp_beta * Q(a,s)) / sum_a' exp(exp_beta*Q(a',s)
            distribution = softmax(action_probabilities, self.exp_beta)
            #torch.multinomial input with num_envs rows, output (num_envs X num_samples) , where num_samples =1
            # we get back a Matrix with size (num_envs x 1)
            actions = torch.multinomial(distribution, 1)

            distribution = distribution.cpu().detach().numpy()

            #transform action matrix to vector
            actions = actions.squeeze().cpu().detach().numpy()
            # TODO: maybe rewrite to tensor if performance drops
            # calculating the log_ratio from pi(a|s)/p(a)
            # -> log(pi(a|s)/p(a))
            # Since distribution, action_prob and actions have #num_envs entries
            # we need a small workaround to return the log ratio also as vector with len(num_envs)
            log_ratio = distribution / action_prob
            log_ratio2 = [log_ratio[i][actions[i]] for i in range(self.num_envs)]
            log_ratio2 = np.log(log_ratio2)
            return actions, log_ratio2

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def change_and_get_beta(self):
        beta = []
        for i in range(self.num_envs):
            if self.exp_beta < self.exp_beta_mid:
                self.exp_beta += self.delta_exp_beta
            elif self.exp_beta >= self.exp_beta_mid:
                self.exp_beta += self.delta_exp_beta2
            elif self.exp_beta > self.exp_beta_end:
                self.exp_beta = self.exp_beta_end
            beta.append(1/self.exp_beta)

        return beta

def softmax(action_prob, beta):
    """
        Function to calc Softmax/Boltzmann-Distribution
    """
    exp = torch.exp(beta * action_prob)
    smax = exp / torch.sum(exp)
    # assert torch.sum(smax) == 1.0
    return smax
