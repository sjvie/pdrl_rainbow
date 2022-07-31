import torch
from torch.nn import functional as F


def get_huber_loss(agent, states, actions, rewards, n_next_states, dones):
    batch_indices = torch.arange(agent.batch_size, device=agent.device).long()

    with torch.no_grad():
        # compute next Q-value using target_network
        q_next = agent.model(n_next_states)

        # take action with highest q_value, _ gets the indices of the max value
        a_star = torch.argmax(q_next, dim=-1)

        if agent.use_double:
            q_target_next = agent.target_model(n_next_states)

            q_a_star = q_target_next[batch_indices, a_star]
        else:
            q_a_star = q_next[batch_indices, a_star]

        target_q_values = rewards + (1 - dones) * agent.discount_factor * q_a_star

    q = agent.model(states)
    current_q_values = q[batch_indices, actions]

    # use Huberloss for error clipping, prevents exploding gradients
    loss = F.huber_loss(current_q_values, target_q_values, reduction="none")

    td_errors = target_q_values - current_q_values

    priorities = abs(td_errors).clamp(min=agent.replay_buffer_prio_offset)

    return loss, priorities


def get_distributional_loss(agent, states, actions, rewards, n_next_states, dones):
    # initialize target distribution matrix
    m = torch.zeros((agent.batch_size, agent.num_atoms), device=agent.device)
    batch_indices = torch.arange(agent.batch_size, device=agent.device).long()

    with torch.no_grad():
        # output of online model for n next states
        q_next_dist = agent.model(n_next_states)
        q_next = (q_next_dist * agent.z_support).sum(-1)

        # get best actions for next states according to online model
        # a* = argmax_a(sum_i(z_i *p_i(x_{t+1},a)))
        a_star = torch.argmax(q_next, dim=-1)

        if agent.use_double:
            # output of target model for n next states
            q_target_next_dist = agent.target_model(n_next_states)

            # get distributions for action a* selected by online model
            next_dist = q_target_next_dist[batch_indices, a_star]
        else:
            next_dist = q_next_dist[batch_indices, a_star]

        # Tz = r + gamma*(1-done)*z
        T_z = rewards.unsqueeze(-1) + (1 - dones).unsqueeze(-1) * (
                agent.discount_factor ** agent.n_step_returns) * agent.z_support

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
    q_dist_log = agent.model(states, log=True)
    q_dist_log_a = q_dist_log[batch_indices, actions]

    if agent.use_kl_loss:
        # get Kullbeck-Leibler divergence of target and approximating distribution
        # the KL divergence calculation has some issues as parts of m can be 0.
        # this makes the log(m) = -inf and loss = nan

        # KL divergence does not work when values of the distribution are 0
        m = m.clamp(min=1e-5)
        m /= m.sum(dim=-1, keepdim=True)

        loss = (m * m.log() - m * q_dist_log_a).sum(dim=-1)  # KL divergence
    else:
        loss = -torch.sum(m * q_dist_log_a, dim=-1)  # cross entropy

    # todo: remove
    if torch.isnan(loss).any() or (loss < 0).any():
        torch.set_printoptions(profile="full")
        print("loss:", loss)
        print("m:", m)
        print("q_next_dist:", q_next_dist)
        print("q_next:", q_next)
        print("z_support:", agent.z_support)
        print("a_star:", a_star)
        print("next_dist:", next_dist)
        print("T_z:", T_z)
        print("bj:", bj)
        print("l:", l)
        print("u:", u)
        print("l_add:", l_add)
        print("u_add:", u_add)
        print("same_add:", same_add)
        print("q_dist_log:", q_dist_log)
        print("q_dist_log_a:", q_dist_log_a)
        torch.set_printoptions(profile="default")
        if torch.isnan(loss).any():
            assert False, "here you go ..."

    priorities = loss.clamp(min=agent.replay_buffer_prio_offset)

    return loss, priorities
