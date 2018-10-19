import torch
# requires more recent version of torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, device):
        self.clear_history()
        self.device = device

    def insert(self, obs, action, r, value, action_logprob, mask):
        self.observations.append(obs.clone())
        self.actions.append(action)
        self.rewards.append(r)
        self.value_preds.append(value)
        self.action_logprobs.append(action_logprob)
        self.masks.append(mask)

    def clear_history(self):
        self.observations = []
        self.actions = []
        self.action_logprobs = []
        self.value_preds = []
        self.rewards = []
        self.masks = []
        self.returns = []
        
    def convert_to_tensor(self):
        self.observations = torch.cat(self.observations).to(self.device)
        self.actions = torch.cat(self.actions).to(self.device)
        self.returns = torch.cat(self.returns).to(self.device)
        self.value_preds = torch.cat(self.value_preds).to(self.device)
        self.action_logprobs = torch.cat(self.action_logprobs).to(self.device)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        self.returns = [0]*len(self.rewards)
        if use_gae:
            self.value_preds.append(next_value)
            gae = 0
            for step in reversed(range(len(self.rewards))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
            self.value_preds = self.value_preds[:-1]

        else:
            self.returns.append(next_value)
            for step in reversed(range(len(self.rewards))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]
            self.returns = self.returns[:-1]

    def feed_forward_generator(self, advantages, num_mini_batch):
        batch_size = len(self.rewards)
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices).to(self.device)
            observations_batch = self.observations[indices]
            actions_batch = self.actions[indices]
            return_batch = self.returns[indices]
            old_action_log_probs_batch = self.action_logprobs[indices]
            adv_targ = advantages[indices]
            yield observations_batch, actions_batch, return_batch, old_action_log_probs_batch, adv_targ

    @property              
    def size(self):
        return len(self.rewards)             