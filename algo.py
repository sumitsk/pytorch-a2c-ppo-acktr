import torch
import torch.nn as nn


def update(agent):
    if agent.args.algo == 'a2c':
        dist_entropy, value_loss, action_loss = a2c_update(agent)
    elif agent.args.algo == 'ppo':
        dist_entropy, value_loss, action_loss = ppo_update(agent)
    elif agent.args.algo == 'acktr':
        dist_entropy, value_loss, action_loss = acktr_update(agent)     

    return dist_entropy, value_loss, action_loss
    
        
def ppo_update(agent):
    advantages = agent.rollouts.returns - agent.rollouts.value_preds
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    for e in range(agent.args.ppo_epoch):
        data_generator = agent.rollouts.feed_forward_generator(advantages, agent.args.num_mini_batch)

        for sample in data_generator:
            observations_batch, actions_batch, return_batch, old_action_log_probs_batch, adv_targ = sample
            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = agent.actor_critic.evaluate_actions(observations_batch, actions_batch)

            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - agent.args.clip_param, 1.0 + agent.args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
            value_loss = (return_batch - values).pow(2).mean()

            agent.optimizer.zero_grad()
            (value_loss + action_loss - dist_entropy * agent.args.entropy_coef).backward()
            nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), agent.args.max_grad_norm)
            agent.optimizer.step()

    return dist_entropy, value_loss, action_loss        

def a2c_update(agent):
    values, action_log_probs, dist_entropy, states = agent.actor_critic.evaluate_actions(agent.rollouts.observations[:-1].view(-1, *agent.obs_shape),
                                                                                         agent.rollouts.states[0].view(-1, agent.actor_critic.state_size),
                                                                                         agent.rollouts.masks[:-1].view(-1, 1),
                                                                                         agent.rollouts.actions.view(-1, agent.action_shape))

    values = values.view(agent.args.num_steps, agent.args.num_processes, 1)
    action_log_probs = action_log_probs.view(agent.args.num_steps, agent.args.num_processes, 1)
    advantages = agent.rollouts.returns[:-1] - values
    value_loss = advantages.pow(2).mean()
    action_loss = -(advantages.detach() * action_log_probs).mean()

    agent.optimizer.zero_grad()
    total_loss = value_loss * agent.args.value_loss_coef + action_loss - dist_entropy * agent.args.entropy_coef
    total_loss.backward()
    nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), agent.args.max_grad_norm)
    agent.optimizer.step()

    return dist_entropy, value_loss, action_loss


def acktr_update(agent):
    values, action_log_probs, dist_entropy, states = agent.actor_critic.evaluate_actions(agent.rollouts.observations[:-1].view(-1, *agent.obs_shape),
                                                                                         agent.rollouts.states[0].view(-1, agent.actor_critic.state_size),
                                                                                         agent.rollouts.masks[:-1].view(-1, 1),
                                                                                         agent.rollouts.actions.view(-1, agent.action_shape))

    values = values.view(agent.args.num_steps, agent.args.num_processes, 1)
    action_log_probs = action_log_probs.view(agent.args.num_steps, agent.args.num_processes, 1)
    advantages = agent.rollouts.returns[:-1] - values
    value_loss = advantages.pow(2).mean()
    action_loss = -(advantages.detach() * action_log_probs).mean()

    if agent.optimizer.steps % agent.optimizer.Ts == 0:
        # Sampled fisher, see Martens 2014
        agent.actor_critic.zero_grad()
        pg_fisher_loss = -action_log_probs.mean()

        value_noise = torch.randn(values.size()).to(advantages.device) 
        sample_values = values + value_noise
        vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

        fisher_loss = pg_fisher_loss + vf_fisher_loss
        agent.optimizer.acc_stats = True
        fisher_loss.backward(retain_graph=True)
        agent.optimizer.acc_stats = False

    agent.optimizer.zero_grad()
    total_loss = value_loss * agent.args.value_loss_coef + action_loss - dist_entropy * agent.args.entropy_coef
    total_loss.backward()
    agent.optimizer.step()
    
    return dist_entropy, value_loss, action_loss