import torch
from torch.nn import functional as F


def get_huber_loss(agent, states, actions, rewards, n_next_states, dones):
    with torch.no_grad():
        # compute next Q-value using target_network
        q_online_next = agent.target_model(n_next_states, dist=False, z_support=agent.z_support)

        # take action with highest q_value, _ gets the indices of the max value
        a_star = torch.argmax(q_online_next, dim=-1)
        #next_q_values_a, _ = next_q_values.max(dim=1)

        q_target = agent.target_model(n_next_states, dist=False, z_support=agent.z_support)
        q_target_a_star = q_target[range(agent.batch_size), a_star]

        #rewards = rewards.unsqueeze(-1)
        target_q_values = rewards + (1 - dones) * agent.discount_factor * q_target_a_star

    q_online = agent.online_model(states, dist=False, z_support=agent.z_support)
    current_q_values = q_online[range(agent.batch_size), actions]

    # use Huberloss for error clipping, prevents exploding gradients
    loss = F.huber_loss(current_q_values, target_q_values, reduction="none")

    td_error = current_q_values - (rewards + agent.discount_factor * target_q_values)
    priorities = abs(td_error).clamp(min=agent.replay_buffer_prio_offset)

    return loss, priorities


def get_distributional_loss(agent, states, actions, rewards, n_next_states, dones):
    # initialize target distribution matrix
    m = torch.zeros(agent.batch_size, agent.num_atoms, device=agent.device)

    with torch.no_grad():
        # output of online model for n next states
        q_online = agent.online_model(n_next_states, dist=False, z_support=agent.z_support)

        # get best actions for next states according to online model
        # a* = argmax_a(sum_i(z_i *p_i(x_{t+1},a)))
        a_star = torch.argmax(q_online, dim=-1)

        # Double DQN part
        # output of target model for n next states
        q_target = agent.target_model(n_next_states, dist=True)

        # get distributions for action a* selected by online model
        next_dist = q_target[range(agent.batch_size), a_star]

        # Tz = r + gamma*(1-done)*z
        T_z = rewards.unsqueeze(-1) + torch.outer(1 - dones,
                                                  (agent.discount_factor ** agent.n_step_returns) * agent.z_support)

        # eingrenzen der Werte
        T_z = T_z.clamp(min=agent.v_min, max=agent.v_max)

        # bj ist hier der index der atome auf denen die Distribution liegt
        bj = (T_z - agent.v_min) / agent.z_delta

        # l und u sind die ganzzahligen indizes auf die bj projeziert werden soll
        l = bj.floor().long()
        u = bj.ceil().long()

        # values to be added at the l and u indices
        l_add = (u - bj) * next_dist
        u_add = (bj - l) * next_dist

        # values to be added at the indices where l == u == bj
        # todo: is this needed? It does not seem to be a part of the algorithm in the dist paper
        same_add = (u == l) * next_dist

        # add values to m at the given indices
        m.view(-1).index_add_(0, u.view(-1) + agent.index_offset, u_add.view(-1))
        m.view(-1).index_add_(0, l.view(-1) + agent.index_offset, l_add.view(-1))
        m.view(-1).index_add_(0, l.view(-1) + agent.index_offset, same_add.view(-1))

    # output of online model for states
    # shape (batch_size, action_space, num_atoms)
    q_dist = agent.online_model(states, dist=True)
    q_dist_a = q_dist[range(agent.batch_size), actions]

    # get Kullbeck-Leibler divergence of target and approximating distribution
    # the KL divergence calculation has some issues as parts of m can be 0.
    # this makes the log(m) = -inf and loss = nan

    # KL divergence does not work when values of the distribution are 0
    # m = m.clamp(min=1e-9)

    # loss = (m * (m / q_dist_a).log()).sum(dim=-1)  # KL divergence

    # loss = torch.sum(m * torch.log(m) - m * q_dist_a, dim=-1)  # KL divergence (is it though?)
    loss = cross_entropy(m, q_dist_a)  # cross entropy

    assert not (loss < 0).any()
    assert not torch.isnan(loss).any()

    priorities = loss.clamp(min=agent.replay_buffer_prio_offset)

    return loss, priorities


def cross_entropy(target_dist, pred_dist):
    return -torch.sum(target_dist * pred_dist.log(), dim=-1)
