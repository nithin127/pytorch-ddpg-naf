import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

"""
From: https://github.com/pytorch/pytorch/issues/1959
There's an official LayerNorm implementation in pytorch now, but it hasn't been included in 
pip version yet. This is a temporary version
This slows down training by a bit
"""
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space, image_input):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.image_input = image_input
        num_outputs = action_space.shape[0]

        if image_input:
            self.conv1 = nn.Conv2d(num_inputs[2], 32, 8, stride=2)
            self.conv1_drop = torch.nn.Dropout2d(p=0.2)

            self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
            self.conv2_drop = torch.nn.Dropout2d(p=0.2)

            self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
            self.conv3_drop = torch.nn.Dropout2d(p=0.2)

            self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

            self.linear1_drop = nn.Dropout(p=0.5)
            self.linear1 = nn.Linear(32 * 9 * 14, hidden_size)
        else:
            self.linear1 = nn.Linear(num_inputs[0], hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        if self.image_input:
            inputs = inputs.permute((0,3,1,2))
            x = self.conv1(inputs)
            #x = self.conv1_drop(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            #x = self.conv2_drop(x)
            x = F.leaky_relu(x)

            x = self.conv3(x)
            #x = self.conv3_drop(x)
            x = F.leaky_relu(x)

            x = self.conv4(x)
            x = F.leaky_relu(x)

            x = x.view(-1, 32 * 9 * 14)

        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = F.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space, image_input):
        super(Critic, self).__init__()
        self.action_space = action_space
        self.image_input = image_input
        num_outputs = action_space.shape[0]

        if image_input:
            self.conv1 = nn.Conv2d(num_inputs[2], 32, 8, stride=2)
            self.conv1_drop = torch.nn.Dropout2d(p=0.2)

            self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
            self.conv2_drop = torch.nn.Dropout2d(p=0.2)

            self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
            self.conv3_drop = torch.nn.Dropout2d(p=0.2)

            self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

            self.linear1_drop = nn.Dropout(p=0.5)
            self.linear1 = nn.Linear(32 * 9 * 14, hidden_size)
        else:
            self.linear1 = nn.Linear(num_inputs[0], hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        if self.image_input:
            inputs = inputs.permute((0,3,1,2))
            x = self.conv1(inputs)
            #x = self.conv1_drop(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            #x = self.conv2_drop(x)
            x = F.leaky_relu(x)

            x = self.conv3(x)
            #x = self.conv3_drop(x)
            x = F.leaky_relu(x)

            x = self.conv4(x)
            x = F.leaky_relu(x)

            x = x.view(-1, 32 * 9 * 14)

        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V

class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, obs_shape, action_space, image_input):

        self.obs_shape = obs_shape
        self.action_space = action_space
        
        self.actor = Actor(hidden_size, self.obs_shape, self.action_space, image_input)
        self.actor_target = Actor(hidden_size, self.obs_shape, self.action_space, image_input)
        self.actor_perturbed = Actor(hidden_size, self.obs_shape, self.action_space, image_input)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.obs_shape, self.action_space, image_input)
        self.critic_target = Critic(hidden_size, self.obs_shape, self.action_space, image_input)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        if param_noise is not None: 
            mu = self.actor_perturbed((Variable(state)))
        else:
            mu = self.actor((Variable(state)))

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return mu.clamp(-1, 1)


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))