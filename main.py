import os
import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from functools import reduce
import operator

import gym
import numpy as np
from gym import wrappers

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

import torch
from ddpg import DDPG
from naf import NAF
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
from logger_sc import Logger

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | NAF')
parser.add_argument('--env-name', default="SimpleSim-v0",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
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
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='number of episodes (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack')    
parser.add_argument('--resume-training', type=bool, default=False,
                    help='To resume training or not')
parser.add_argument('--suffix', default="",
                    help='To resume training or not')
parser.add_argument('--sliding-window-size', type=int, default=30,
                    help='number of values to compute average over')    
args = parser.parse_args()



logger_dir = os.path.join("logs", "{}_{}_{}".format(args.env_name, args.algo, args.suffix))
if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)
logger = Logger(logger_dir)

env = HeadingWrapper(gym.make(args.env_name))
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


memory = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, 
    desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

rewards_train = []
rewards_test = []
total_numsteps = 0
updates = 0
policy_loss_list = []
value_loss_list = []

if args.resume_training:
    end_str = "_{}_{}".format(args.env_name, args.suffix)
    agent.load_model("models/ddpg_actor" + end_str, "models/ddpg_critic" + end_str)

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()]).to(device)
    
    if args.ou_noise: 
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise and args.algo == "DDPG":
        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0
    while True:
        action_noise = torch.Tensor(ounoise.noise()).to(device)
        action = agent.select_action(state, action_noise, param_noise)
        next_state, reward, done, _ = env.step(action.numpy()[0])
        total_numsteps += 1
        episode_reward += reward

        #action = torch.Tensor(action).to(device)
        mask = torch.Tensor([not done]).to(device)
        next_state = torch.Tensor([next_state]).to(device)
        reward = torch.Tensor([reward]).to(device)
        
        memory.push(state, action, mask, next_state, reward)

        state = next_state
        
        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                
                value_loss, policy_loss = agent.update_parameters(batch)
                value_loss_list.append(value_loss)
                policy_loss_list.append(policy_loss)
                updates += 1

                logger.log_scalar_rl("loss/value", value_loss_list, args.sliding_window_size, [i_episode, total_numsteps, updates])
                logger.log_scalar_rl("loss/policy", policy_loss_list, args.sliding_window_size, [i_episode, total_numsteps, updates])                
        if done:
            break

    print("Episode: {}, episode_reward: {}, total_numsteps: {}".format(i_episode, episode_reward, total_numsteps))
    rewards_train.append(episode_reward)
    logger.log_scalar_rl("reward/train", rewards_train, args.sliding_window_size, [i_episode, total_numsteps, updates])
    
    # Update param_noise based on distance metric
    if args.param_noise:
        episode_transitions = memory.memory[memory.position-t:memory.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
        param_noise.adapt(ddpg_dist)

    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()]).to(device)
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward

            next_state = torch.Tensor([next_state]).to(device)
            
            state = next_state
            if done:
                break

        rewards_test.append(episode_reward)
        print("Episode: {}, total numsteps: {}, test_reward: {}, test_average reward: {}".format(i_episode, total_numsteps, rewards_test[-1], np.mean(rewards_test[-10:])))

        logger.log_scalar_rl("reward/test", rewards_test, args.sliding_window_size, [i_episode, total_numsteps, updates])
        agent.save_model(args.env_name, args.suffix)

        
env.close()
