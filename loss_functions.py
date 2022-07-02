import torch
from torch.nn import functional as F


def get_huber_loss(agent, states, actions, rewards, n_next_states, dones):
    with torch.no_grad():
        # compute next Q-value using target_network
        next_q_values = agent.target_model(n_next_states)

        # take action with highest q_value, _ gets the indices of the max value
        next_q_values, _ = next_q_values.max(dim=1)

        # avoid broadcast issue /just to be sure
        next_q_values = next_q_values.reshape(-1, 1)

        rewards = rewards.unsqueeze(-1)
        target_q_values = rewards + (1 - dones.unsqueeze(1)) * agent.discount_factor * next_q_values

    current_q_values = agent.online_model(states)
    # ToDO: why does this work?!?
    current_q_values = current_q_values.gather(1, actions.unsqueeze(-1))

    # use Huberloss for error clipping, prevents exploding gradients
    loss = F.huber_loss(current_q_values, target_q_values, reduction="none")
    return loss


def get_distributional_loss(agent, states, actions, rewards, n_next_states, dones):
    # initialize target distribution matrix
    m = torch.zeros(agent.batch_size, agent.num_atoms, device=agent.device)

    # logarithmic output of online model for states
    # shape (batch_size, action_space, num_atoms)
    log_q_dist = agent.online_model.forward(states, log=True)
    log_q_dist_a = log_q_dist[range(agent.batch_size), actions]

    with torch.no_grad():
        # non-logarithmic output of online model for n next states
        q_online = agent.online_model(n_next_states)

        # get best actions for next states according to online model
        # a* = argmax_a(sum_i(z_i *p_i(x_{t+1},a)))
        a_star = torch.argmax((q_online * agent.z_support).sum(-1), dim=1)

        # Double DQN part
        # non-logarithmic output of target model for n next states
        q_target = agent.target_model.forward(n_next_states)

        # get distributions for action a* selected by online model
        next_dist = q_target[range(agent.batch_size), a_star]

        # Tz = r + gamma*(1-done)*z
        T_z = rewards.unsqueeze(-1) + torch.outer(agent.discount_factor ** agent.n_step_returns * (1 - dones),
                                                  agent.z_support)

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

    # get Kullbeck-Leibler divergence of target and approximating distribution
    # the KL divergence calculation has some issues as parts of m can be 0.
    # this makes the log(m) = -inf and loss = nan
    # loss = torch.sum(m * torch.log(m) - m * log_q_dist, dim=-1) # KL divergence
    loss = - torch.sum(m * log_q_dist_a, dim=-1)  # cross entropy
    return loss
