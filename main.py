#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gym
from agent.dqn_agent import DQNAgent
from utils.network_architecture import CartpoleNetwork
from utils.network_architecture import AtariNetwork
from utils.wrapper import make_atari
from utils.wrapper import PongWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_atari("PongNoFrameskip-v4")

network = AtariNetwork(n_actions=6)
agent = DQNAgent(device, 6, network, epsilon_decay=20000)
agent.train(env, 15)