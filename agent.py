import copy
import os
import time
import numpy as np
import pdb

import torch
import torch.optim as optim
from torch.autograd import Variable

from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from utils import save_checkpoint
from algo import update
from arguments import get_args

#import rllab.misc.logger as logger
from rllab.envs.normalized_env import normalize

from tensorboardX import SummaryWriter

# if using gpu
# UGLY WAY TO CALL ARGUMENT IN EVERY FILE
args = get_args()
use_cuda = args.cuda
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Agent(object):
    def __init__(self, args):
        self.args = args

        if self.args.env_name == 'ant':
            from rllab.envs.mujoco.ant_env import AntEnv
            env = AntEnv()
            # set the target velocity direction (for learning sub-policies)
            env.velocity_dir = self.args.velocity_dir
            env.penalty = self.args.penalty
            
            # use gym environment observation 
            env.use_gym_obs = self.args.use_gym_obs
            # use gym environment reward
            env.use_gym_reward = self.args.use_gym_reward
            
        elif self.args.env_name == 'swimmer':
            from rllab.envs.mujoco.swimmer_env import SwimmerEnv
            env = SwimmerEnv()  
            env.velocity_dir = self.args.velocity_dir
        else:
            raise NotImplementedError    

        self.env = normalize(env) 
        self.reset_env()

        self.obs_shape = self.env.observation_space.shape
        
        self.actor_critic = self.select_network()
        self.optimizer = self.select_optimizer()
        if self.args.cuda:
            self.actor_critic.cuda()

        # list of RolloutStorage objects
        self.episodes_rollout = []      
        # concatenation of all episodes' rollout
        self.rollouts = RolloutStorage()    
        # this directory is used for tensorboardX only
        self.writer = SummaryWriter(args.log_dir+self.args.velocity_dir)

        self.episodes = 0
        self.episode_steps = []
        self.train_rewards = []
        
    def select_network(self):
        actor_critic = MLPPolicy(self.obs_shape[0], self.env.action_space)
        return actor_critic

    def select_optimizer(self):
        if self.args.algo == 'acktr':
            optimizer = KFACOptimizer(self.actor_critic)
        elif self.args.use_adam:
            optimizer = optim.Adam(self.actor_critic.parameters(),
                                   self.args.lr)
        else:
            optimizer = optim.RMSprop(self.actor_critic.parameters(),
                                      self.args.lr,
                                      eps=self.args.eps,
                                      alpha=self.args.alpha)
        return optimizer

    def reset_env(self):
        self.current_obs = Tensor(self.env.reset()).view(1, -1)

    def custom_reward(self, info):
        raise NotImplementedError

    def load(self, filename):
        # load models and optimizer
        #'''
        load_device = 'cuda:0'
        target_device = 'cpu'
        #target_device = 'cuda:0' if use_cuda else 'cpu'
        '''
        load_device = 'cpu'
        target_device = 'cpu'
        '''  
        checkpoint = torch.load(filename, map_location={load_device: target_device})
        self.actor_critic.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        episode_number = checkpoint['episode_number']
        return episode_number

    def save(self, episode_num, filename):
        # save models and optimizer at checkpoint
        save_checkpoint({
            'episode_number': episode_num,
            'state_dict': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename=filename)


    # rollout one episode
    def rollout_episode(self, test=False, render=False):
        rollout = RolloutStorage()
        self.reset_env()
        step = 0
        done = False
        while not done:
            step += 1
            value, action, action_logprob = self.actor_critic.act(
                                Variable(self.current_obs, volatile=True),
                                deterministic=test==True
                                )

            cpu_actions = action.data.squeeze(1).cpu().numpy()[0]
            next_obs, reward, done, info = self.env.step(cpu_actions)
            next_obs = Tensor(next_obs).view(1, -1)
            
            if render:
                self.env.render()
        
            # a constant reward scaling factor can be introduced to stabilise training and prevent large value losses
            r = reward * self.args.reward_scale
            done = done or step == self.args.episode_max_length
            mask = 1.0 if not done else 0.0
            rollout.insert(self.current_obs, action.data, r, 
                           value.data, action_logprob.data, mask)
            self.current_obs.copy_(next_obs)
        
        if not test:
                next_value = self.actor_critic.forward(
                                Variable(rollout.observations[-1], volatile=True)
                                )[0].data
                rollout.compute_returns(next_value, self.args.use_gae,
                                        self.args.gamma, self.args.tau)

                self.episode_steps.append(step)
                
        return rollout
        
    def pre_update(self):
        self.concatenate_rollouts()    
        self.rollouts.convert_to_tensor()

    def post_update(self):
        self.episodes_rollout = []
        self.rollouts.clear_history()  

    def concatenate_rollouts(self):
        assert self.rollouts.size == 0, 'ROLLOUT IS NOT EMPTY'
        for rollout in self.episodes_rollout:
            self.rollouts.observations += rollout.observations
            self.rollouts.actions += rollout.actions
            self.rollouts.action_logprobs += rollout.action_logprobs
            self.rollouts.value_preds += rollout.value_preds
            self.rollouts.rewards += rollout.rewards
            self.rollouts.returns += rollout.returns
            
    def collect_samples(self, num_episodes):
        for ep in range(num_episodes):
            rollout = self.rollout_episode()
            self.episodes_rollout.append(rollout)
            self.episodes += 1
            episode_reward = np.sum(rollout.rewards)
            self.train_rewards.append(episode_reward)
            self.log_to_tensorboard(rollout)
    
    def log_to_tensorboard(self, rollout):
        self.writer.add_scalar('train_reward_'+self.args.velocity_dir, 
            np.sum(rollout.rewards), self.episodes)        

    def train(self):
        start = time.time()
        j = 0
        while j < self.args.num_updates:
            j += 1

            self.collect_samples(self.args.update_frequency)
            self.pre_update()
            dist_entropy, value_loss, action_loss = update(self)
            self.post_update()

            #print('\nEpisode: %d Reward %4f' %(self.episodes, self.train_rewards[-1]))
            print('\nEpisode: %d' %(self.episodes))
            print('Steps: %d' %(self.episode_steps[-1]))
            print('Reward: %4f' %(self.train_rewards[-1]))
            print('Value Loss: %4f' %(value_loss))
            print('Action Loss: %4f' %(action_loss))
            print('Dist Entropy: %4f' %(dist_entropy))

            if j % self.args.save_interval == 0 and self.args.save_dir != "":
                episode_num = j * self.args.update_frequency
                filename = self.args.save_dir + self.args.env_name + '_' + \
                           str(episode_num) + '.pt'
                self.save(episode_num, filename)

            if j % self.args.eval_interval == 0:
                test_reward_mean, test_reward_std = self.eval_model(num_episodes=20)
                self.writer.add_scalar('test_reward_'+self.args.velocity_dir,
                                        test_reward_mean, self.episodes)

    def eval_model(self, num_episodes):
        rewards = []
        for i in range(num_episodes):
            rollout = self.rollout_episode(test=True, render=False)
            rewards.append(np.sum(rollout.rewards))
        return np.mean(rewards), np.std(rewards)            


    def test(self, num_episodes=100):
        self.load(self.args.model_filename)
        for i in range(num_episodes):
                rollout = self.rollout_episode(test=True, render=True)
                print('Episode %d Reward: %4f' %(i+1, np.sum(rollout.rewards)))
