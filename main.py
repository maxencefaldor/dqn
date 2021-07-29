#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gym

import matplotlib.pyplot as plt

from agent.dqn_agent import DQNAgent
from agent.d2qn_agent import D2QNAgent
from utils.network_architecture import CartpoleNetwork
from utils.network_architecture import AtariNetwork
from utils.network_architecture import DuelingCartpoleNetwork
from utils.wrapper import make_atari
from utils.wrapper import make_cartpole_swing_up

from replay_memory.replay_buffer import ReplayBuffer
from replay_memory.prioritized_replay_buffer import PrioritizedReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = make_atari("PongNoFrameskip-v4", {0:4, 1:5})
#env = make_cartpole_swing_up("CartPole-v1", max_episode_steps=1000)
env = gym.make("CartPole-v1")

import copy
network_1 = DuelingCartpoleNetwork()
network_2 = copy.deepcopy(network_1)

agent_1 = DQNAgent(device,
                  2,
                  network_1,
                  lr=0.0005,
                  gamma=0.99,
                  n=4,
                  batch_size=32,
                  n_gradient_steps=4,
                  epsilon_min=0.01,
                  epsilon_decay=2000,
                  buffer_size=1e6,
                  beta=100)
return_list_1 = agent_1.train(env, 100)

# agent_2 = D2QNAgent(device,
#                   2,
#                   network_2,
#                   lr=0.0005,
#                   gamma=0.99,
#                   n=4,
#                   batch_size=32,
#                   n_gradient_steps=4,
#                   epsilon_min=0.01,
#                   epsilon_decay=2000,
#                   buffer_size=1e6,
#                   beta=100)
# return_list_2 = agent_2.train(env, 100)

plt.plot(return_list_1, label="Agent 1")
# plt.plot(return_list_2, label="Agent 2")
plt.legend()
