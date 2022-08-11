import torch
import torch.nn as nn
import numpy as np
from model import RewardModel
class RND:
    def __init__(self, config,action_space,linear_layer,in_channels):
        self.reward_target = RewardModel(conf=config,
                                          action_space=action_space,
                                          linear_layer=linear_layer,
                                          in_channels=in_channels
                                          )
        self.reward_predicator = RewardModel(conf=config,
                                          action_space=action_space,
                                          linear_layer=linear_layer,
                                          in_channels=in_channels
                                          )
        self.optimizer = torch.optim.Adam(self.reward_predicator.parameters(),
                                          lr=config.adam_learning_rate,
                                          eps=config.adam_e)
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1,1,84,84))

    def train(self,next_obs):
        next_obs = next_obs.cpu().numpy()
        next_obs = self.norm_obs(next_obs)
        loss = self.estimate(next_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # estimate intrinsic reward by using the mse between target and predictor values
    def estimate(self, next_obs):
        target_val = self.reward_target(next_obs)
        pred_val = self.reward_predicator(next_obs)
        r_i = nn.functional.mse_loss(target_val.detach(),pred_val,reduction='none').mean(dim=-1)
        return r_i

    # calculate normalized intrinsic reward
    def calc_int(self,next_obs):
        obs = self.norm_obs(next_obs)
        ints = self.estimate(obs)
        mean, std, count = self.calc_mean_std_count(ints)
        self.reward_rms.update_from_moments(mean,std**2,count)
        ints = ints / np.sqrt(self.reward_rms.var)
        return ints
    def update_obs(self,next_obs):
        self.obs_rms.update(next_obs)

    def norm_obs(self,next_obs):
        obs = ((next_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5,5)
        return obs

    def calc_mean_std_count(self,rewards):
        return np.mean(rewards), np.std(rewards), len(rewards)



class RunningMeanStd(object):
    # original: https://github.com/jcwleo/random-network-distillation-pytorch
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # shape of (1,1,84,84) ensures that we can calc mean & var over first dimension and still add the "full"
    # observations to count
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
