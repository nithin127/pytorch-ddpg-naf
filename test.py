import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter
from functools import reduce
import operator

import gym
import numpy as np
from gym import wrappers

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | NAF')
parser.add_argument('--env-name', default="SimpleSim-v0",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack')    
parser.add_argument('--model-suffix', default="",
                    help='To resume training or not')
args = parser.parse_args()

env = NormalizedActions(gym.make(args.env_name))

writer = SummaryWriter()

env.seed(args.seed)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

np.random.seed(args.seed)

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

if len(env.observation_space.shape) == 3:
    image_input = True

if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.hidden_size,
                      obs_shape, env.action_space, image_input)
    if torch.cuda.device_count() > 1:
        agent.model = nn.DataParallel(agent.model)
        agent.target_model = nn.DataParallel(agent.target_model)
    agent.model.to(device)
    agent.target_model.to(device)
else:
    agent = DDPG(args.gamma, args.tau, args.hidden_size,
                      obs_shape, env.action_space, image_input)
    if torch.cuda.device_count() > 1:
        import torch.nn as nn
        agent.actor = nn.DataParallel(agent.actor)
        agent.actor_target = nn.DataParallel(agent.actor_target)
        agent.actor_perturbed = nn.DataParallel(agent.actor_perturbed)
        agent.critic = nn.DataParallel(agent.critic)
        agent.critic_target = nn.DataParallel(agent.critic_target)
    agent.actor.to(device)
    agent.actor_target.to(device)
    agent.actor_perturbed.to(device)
    agent.critic.to(device)
    agent.critic_target.to(device)


end_str = "_{}_{}".format(args.env_name, args.model_suffix)
agent.load_model("models/ddpg_actor" + end_str, "models/ddpg_critic" + end_str)

while True:
    episode_reward = 0
    state = torch.Tensor([env.reset()]).to(device)
    env.render()
    while True:
        action = agent.select_action(state, None, None)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        env.render()
        episode_reward += reward

        #action = torch.Tensor(action).to(device)
        mask = torch.Tensor([not done]).to(device)
        next_state = torch.Tensor([next_state]).to(device)
        reward = torch.Tensor([reward]).to(device)
        
        state = next_state
        print("Reward: {}; Episode reward: {}".format(reward, episode_reward))

        if done:
            break

    
env.close()
