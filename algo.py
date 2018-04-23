import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

from adam_new import Adam_Custom
from clip_grad_norm import clip_grad_norm_

def update(agent):
    if agent.args.algo == 'a2c':
        dist_entropy, value_loss, action_loss = a2c_update(agent)
    elif agent.args.algo == 'ppo':
        dist_entropy, value_loss, action_loss = ppo_update(agent)
    elif agent.args.algo == 'acktr':
        dist_entropy, value_loss, action_loss = acktr_update(agent)

    return dist_entropy.data[0], value_loss.data[0], action_loss.data[0]


def ppo_update(agent):
    advantages = agent.rollouts.returns - agent.rollouts.value_preds
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    for e in range(agent.args.ppo_epoch):
        # only feed_forward_generator supported for now
        data_generator = agent.rollouts.feed_forward_generator(
                                    advantages,
                                    agent.args.num_mini_batch)

        for sample in data_generator:
            observations_batch, actions_batch, return_batch, \
                old_action_log_probs_batch, adv_targ = sample

            values, action_log_probs, dist_entropy = \
                    agent.actor_critic.evaluate_actions(
                                        Variable(observations_batch),
                                        Variable(actions_batch)
                                        )

            adv_targ = Variable(adv_targ)
            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - agent.args.clip_param, 1.0 + agent.args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            agent.optimizer.zero_grad()
            (value_loss + action_loss - dist_entropy * agent.args.entropy_coef).backward()
            nn.utils.clip_grad_norm(agent.actor_critic.parameters(), agent.args.max_grad_norm)
            agent.optimizer.step()

    return dist_entropy, value_loss, action_loss


def a2c_update(agent):
    values, action_log_probs, dist_entropy = \
            agent.actor_critic.evaluate_actions(
                Variable(agent.rollouts.observations.view(-1, *agent.obs_shape)),
                Variable(agent.rollouts.actions.view(-1, agent.action_shape))
                )

    advantages = Variable(agent.rollouts.returns) - values
    value_loss = advantages.pow(2).mean()

    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    agent.optimizer.zero_grad()
    total_loss = value_loss * agent.args.value_loss_coef + action_loss - \
                    dist_entropy * agent.args.entropy_coef
    total_loss.backward()
    nn.utils.clip_grad_norm(agent.actor_critic.parameters(), agent.args.max_grad_norm)
    agent.optimizer.step()

    return dist_entropy, value_loss, action_loss


def acktr_update(agent):
    values, action_log_probs, dist_entropy = \
            agent.actor_critic.evaluate_actions(
                Variable(agent.rollouts.observations.view(-1, *agent.obs_shape)),
                Variable(agent.rollouts.actions.view(-1, agent.action_shape))
                )

    advantages = Variable(agent.rollouts.returns) - values
    value_loss = advantages.pow(2).mean()

    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    if agent.optimizer.steps % agent.optimizer.Ts == 0:
        # Sampled fisher, see Martens 2014
        agent.actor_critic.zero_grad()
        pg_fisher_loss = -action_log_probs.mean()

        value_noise = Variable(torch.randn(values.size()))
        if agent.args.cuda:
            value_noise = value_noise.cuda()

        sample_values = values + value_noise
        vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

        fisher_loss = pg_fisher_loss + vf_fisher_loss
        agent.optimizer.acc_stats = True
        fisher_loss.backward(retain_graph=True)
        agent.optimizer.acc_stats = False

    agent.optimizer.zero_grad()
    total_loss = value_loss * agent.args.value_loss_coef + action_loss - \
                    dist_entropy * agent.args.entropy_coef
    total_loss.backward()
    agent.optimizer.step()

    return dist_entropy, value_loss, action_loss


def meta_update(agent,theta_loss,theta_grad):
    advantages = agent.rollouts.returns - agent.rollouts.value_preds
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    for e in range(agent.args.ppo_epoch):
        
        data_generator = agent.rollouts.feed_forward_generator(
                                advantages,
                                agent.args.num_mini_batch)

        for sample in data_generator:
            #Set the weights as theta_task for the forward pass
            set_weights(agent,theta_loss)

            observations_batch, actions_batch, return_batch, \
                old_action_log_probs_batch, adv_targ = sample

            values, action_log_probs, dist_entropy = \
                    agent.actor_critic.evaluate_actions(
                                        Variable(observations_batch),
                                        Variable(actions_batch)
                                        )

            # pdb.set_trace()
            '''
            print('---------------')
            print(return_batch.numpy().squeeze(1))
            print(values.data.numpy().squeeze(1))        
            #print(diff)
            
            pdb.set_trace()
            #'''

            adv_targ = Variable(adv_targ)
            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - agent.args.clip_param, 1.0 + agent.args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            # Set the weights as theta_meta for the backward pass
            set_weights(agent,theta_grad)
            agent.meta_optimizer.zero_grad()
            grads = torch.autograd.grad((value_loss + action_loss - dist_entropy * agent.args.entropy_coef), agent.actor_critic.parameters())
            grads = clip_grad_norm_(grads,agent.args.max_grad_norm)
            agent.meta_optimizer.step(grads)

    return dist_entropy, value_loss, action_loss

def get_weights(agent):
    # state_dicts = {'id': id,
    #              'state_dict': agent.actor_critic.state_dict(),
    #              }

    return agent.actor_critic.state_dict()

def set_weights(agent, state_dicts):
    
    checkpoint = state_dicts

    agent.actor_critic.load_state_dict(checkpoint)