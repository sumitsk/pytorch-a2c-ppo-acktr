import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot
from algo import update

class VecEnvAgent(object):
	def __init__(self, envs, args):
		self.envs = envs
		self.args = args

		obs_shape = self.envs.observation_space.shape
		self.obs_shape = (obs_shape[0] * self.args.num_stack, *obs_shape[1:])
		
		self.actor_critic = self.select_network()
		self.optimizer = self.select_optimizer()	
		if self.args.cuda:	self.actor_critic.cuda()

		self.action_shape = 1 if self.envs.action_space.__class__.__name__ == "Discrete" \
							else self.envs.action_space.shape[0]		
		
		self.current_obs = torch.zeros(self.args.num_processes, *self.obs_shape)
		obs = self.envs.reset()
		self.update_current_obs(obs)
		
		self.rollouts = RolloutStorage(self.args.num_steps, self.args.num_processes, 
			self.obs_shape, self.envs.action_space, self.actor_critic.state_size)
		self.rollouts.observations[0].copy_(self.current_obs)

		# These variables are used to compute average rewards for all processes.
		self.episode_rewards = torch.zeros([self.args.num_processes, 1])
		self.final_rewards = torch.zeros([self.args.num_processes, 1])

		if self.args.cuda:
			self.current_obs = self.current_obs.cuda()
			self.rollouts.cuda()

		if self.args.vis:
			from visdom import Visdom
			self.viz = Visdom(port=args.port)
			self.win = None	

	def select_network(self):
		if len(self.envs.observation_space.shape) == 3:
			actor_critic = CNNPolicy(self.obs_shape[0], self.envs.action_space, 
				self.args.recurrent_policy)
		else:
			assert not self.args.recurrent_policy, \
				"Recurrent policy is not implemented for the MLP controller"
			actor_critic = MLPPolicy(self.obs_shape[0], self.envs.action_space)
			#actor_critic = BPW_MLPPolicy(obs_shape[0], self.envs.action_space)		
		return actor_critic


	def select_optimizer(self):
		if self.args.algo == 'a2c' and not self.args.use_adam:
			optimizer = optim.RMSprop(self.actor_critic.parameters(), self.args.lr, 
				eps=self.args.eps, alpha=self.args.alpha)
		elif self.args.algo == 'ppo' or self.args.algo == 'a2c':
			optimizer = optim.Adam(self.actor_critic.parameters(), self.args.lr, 
				eps=self.args.eps)
		elif self.args.algo == 'acktr':
			optimizer = KFACOptimizer(self.actor_critic)	
		else:
			raise TypeError("Optimizer should be any one from {a2c, ppo, acktr}")	
		return optimizer	


	def update_current_obs(self, obs):
		shape_dim0 = self.envs.observation_space.shape[0]
		obs = torch.from_numpy(obs).float()
		if self.args.num_stack > 1:
			self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
		self.current_obs[:, -shape_dim0:] = obs


	def run(self):
		for step in range(self.args.num_steps):
			value, action, action_log_prob, states = self.actor_critic.act(
				Variable(self.rollouts.observations[step], volatile=True),
				Variable(self.rollouts.states[step], volatile=True),
				Variable(self.rollouts.masks[step], volatile=True)
				)
			cpu_actions = action.data.squeeze(1).cpu().numpy()
			#print (cpu_actions)
			#input()

			# Observe reward and next obs
			obs, reward, done, info = self.envs.step(cpu_actions)
			reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
			self.episode_rewards += reward

			# If done then clean the history of observations.
			masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
			self.final_rewards *= masks
			self.final_rewards += (1 - masks) * self.episode_rewards
			self.episode_rewards *= masks

			if self.args.cuda: masks = masks.cuda()

			if self.current_obs.dim() == 4:
				self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
			else:
				self.current_obs *= masks

			self.update_current_obs(obs)
			self.rollouts.insert(step, self.current_obs, states.data, action.data, 
				action_log_prob.data, value.data, reward, masks)
	
		next_value = self.actor_critic(
						Variable(self.rollouts.observations[-1], volatile=True),
						Variable(self.rollouts.states[-1], volatile=True),
						Variable(self.rollouts.masks[-1], volatile=True)
						)[0].data

		self.rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma, self.args.tau)
		dist_entropy, value_loss, action_loss = update(self)
		self.rollouts.after_update()
		return dist_entropy, value_loss, action_loss

	def train(self, num_updates):
		start = time.time()
		for j in range(num_updates):
			dist_entropy, value_loss, action_loss = self.run()

			if j % self.args.save_interval == 0 and self.args.save_dir != "":
				save_path = os.path.join(self.args.save_dir, self.args.algo)
				try:
					os.makedirs(save_path)
				except OSError:
					pass

				# A really ugly way to save a model to CPU
				save_model = self.actor_critic
				if self.args.cuda:
					save_model = copy.deepcopy(self.actor_critic).cpu()

				save_model = [save_model,
								hasattr(self.envs, 'ob_rms') and self.envs.ob_rms or None]

				torch.save(save_model, os.path.join(save_path, self.args.env_name + ".pt"))

			if j % self.args.log_interval == 0:
				end = time.time()
				total_num_steps = (j + 1) * self.args.num_processes * self.args.num_steps
				print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
					format(j, total_num_steps,
						   int(total_num_steps / (end - start)),
						   self.final_rewards.mean(),
						   self.final_rewards.median(),
						   self.final_rewards.min(),
						   self.final_rewards.max(), dist_entropy.data[0],
						   value_loss.data[0], action_loss.data[0]))
			if self.args.vis and j % self.args.vis_interval == 0:
				try:
					# Sometimes monitor doesn't properly flush the outputs
					self.win = visdom_plot(self.viz, self.win, self.args.log_dir, 
						self.args.env_name, self.args.algo)
				except IOError:
					pass
				
