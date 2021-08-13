#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gym

import matplotlib.pyplot as plt

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.rainbow_agent import RainbowAgent

from utils.network_architectures import Network
from utils.network_architectures import DuelingNetwork
from utils.network_architectures import RainbowNetwork

from utils.wrappers import make_atari
from utils.wrappers import make_cartpole_swing_up


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_cartpole_swing_up("CartPole-v1", max_episode_steps=1000)
n_episodes = 100
n_features = 4
n_actions = 2
n_neurons = 256

# Vanilla DQN
network_1 = Network(n_features=n_features,
                    n_actions=n_actions,
                    n_neurons=n_neurons)
agent_1 = DQNAgent(device,
                   n_actions,
                   network_1,
                   lr=0.0005,
                   gamma=0.99,
                   n=1,
                   batch_size=32,
                   n_gradient_steps=1,
                   epsilon_min=0.01,
                   epsilon_decay=5000,
                   buffer_size=1e6,
                   beta=100,
                   per=False)

# Prioritized DDQN with multi-step bootstrapping
network_2 = DuelingNetwork(n_features=n_features,
                           n_actions=n_actions,
                           n_neurons=n_neurons)
agent_2 = DDQNAgent(device,
                    n_actions,
                    network_2,
                    lr=0.0005,
                    gamma=0.99,
                    n=4,
                    batch_size=32,
                    n_gradient_steps=1,
                    epsilon_min=0.01,
                    epsilon_decay=5000,
                    buffer_size=1e6,
                    beta=100,
                    per=True)

# Rainbow
network_3 = RainbowNetwork(n_features=n_features,
                           n_actions=n_actions,
                           n_neurons=n_neurons,
                           n_atoms=51,
                           v_min=-10,
                           v_max=10)
agent_3 = RainbowAgent(device,
                       n_actions,
                       network_3,
                       lr=0.0005,
                       n_atoms=51,
                       v_min=-10,
                       v_max=10,
                       gamma=0.99,
                       n=4,
                       batch_size=32,
                       n_gradient_steps=1,
                       epsilon_min=0.01,
                       epsilon_decay=5000,
                       buffer_size=1e6,
                       beta=100,
                       per=True,
                       noisy=True)

# Training
step_list_1, return_list_1 = agent_1.train(env, n_episodes)
step_list_2, return_list_2 = agent_2.train(env, n_episodes)
step_list_3, return_list_3 = agent_3.train(env, n_episodes)

# Plot
def mean_window(return_list, window=10):
    return [sum(return_list[i:i+window])/window
            for i in range(len(return_list) - window)]

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(mean_window(return_list_1), color='lightgray', label="DQN")
ax.plot(mean_window(return_list_2), color='dodgerblue',
        label="Prioritized DDQN with multi-step bootstrapping")
ax.plot(mean_window(return_list_3), color='black', label="Rainbow")

plt.legend()
plt.title("Learning curve for CartPole Swing Up")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.show()
