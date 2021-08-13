#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of Q-network architectures.

Examples of network architectures suited for CartPole and Atari 2600
environments. It includes normal, dueling, noisy and distributional networks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Implementation of the noisy linear layer as described in "Noisy Networks
    for Exploration", Fortunato et al. (2017). Noisy linear layers can be used
    in a network architecture to drive exploration instead of the conventional
    epsilon-greedy heuristic.
    """
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init * mu_range)
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init * mu_range)
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        epsilon = torch.randn(size)
        epsilon = epsilon.sign().mul(epsilon.abs().sqrt())
        return epsilon


class Network(nn.Module):
    """Network architecture suited for classic control environments."""
    
    def __init__(self, n_features, n_actions, n_neurons=16):
        """Creates the layers.
        
        Args:
            n_features: int, number of features of the state.
            n_actions: int, number of actions possible for the agent.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AtariNetwork(nn.Module):
    """Network architecture suited for Atari 2600 environment."""
    
    def __init__(self, n_actions):
        """Creates the layers.
        
        Args:
            n_actions: int, number of actions possible for the agent.
        """
        super(AtariNetwork, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class DuelingNetwork(nn.Module):
    """Dueling network architecture suited for classic control environments."""
    
    def __init__(self, n_features, n_actions, n_neurons=16):
        """Creates the layers.
        
        Args:
            n_features: int, number of features of the state.
            n_actions: int, number of actions possible for the agent.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(DuelingNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.value_stream = nn.Sequential(nn.Linear(n_neurons, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(n_neurons, n_actions))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())

class DuelingAtariNetwork(nn.Module):
    """Dueling network architecture suited for Atari 2600 environment."""
    
    def __init__(self, n_actions):
        """Creates the layers.
        
        Args:
            n_actions: int, number of actions possible for the agent.
        """
        super(DuelingAtariNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())
        self.value_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())


class C51Network(nn.Module):
    """C51 network architecture suited for classic control environments."""
    
    def __init__(self, n_features, n_actions, n_neurons=16,
                 n_atoms=51, v_min=-10, v_max=10):
        """Creates the layers.
        
        Args:
            n_features: int, number of features of the state.
            n_actions: int, number of actions possible for the agent.
            n_neurons: int, number of neurons of the hidden layers.
            n_atoms: int, number of atoms for discretization of the support.
            v_min: float, the value distribution support is [v_min, v_max].
            v_max: float, the value distribution support is [v_min, v_max].
        """
        super(C51Network, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_actions*n_atoms)
    
    def forward(self, x):
        distribution = self.distribution(x)
        return torch.sum(self.support*distribution, dim=2)
    
    def logits(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_actions, self.n_atoms)

    def distribution(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=2)

class C51AtariNetwork(nn.Module):
    """C51 network architecture suited for Atari 2600 environment."""
    
    def __init__(self, n_actions, n_atoms=51, v_min=-10, v_max=10):
        """Creates the layers.
        
        Args:
            n_actions: int, number of actions possible for the agent.
            n_atoms: int, number of atoms for discretization of the support.
            v_min: float, the value distribution support is [v_min, v_max].
            v_max: float, the value distribution support is [v_min, v_max].
        """
        super(C51AtariNetwork, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions*n_atoms)
    
    def forward(self, x):
        distribution = self.distribution(x)
        return torch.sum(self.support*distribution, dim=2)
    
    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x.view(-1, self.n_actions, self.n_atoms)
    
    def distribution(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=2)


class RainbowNetwork(nn.Module):
    """Rainbow network architecture suited for classic control environments.
    
    Specifically, it is a dueling, noisy, distributional architecture."""
    
    def __init__(self, n_features, n_actions, n_neurons=16,
                 n_atoms=51, v_min=-10, v_max=10):
        """Creates the layers.
        
        Args:
            n_features: int, number of features of the state.
            n_actions: int, number of actions possible for the agent.
            n_neurons: int, number of neurons of the hidden layers.
            n_atoms: int, number of atoms for discretization of the support.
            v_min: float, the value distribution support is [v_min, v_max].
            v_max: float, the value distribution support is [v_min, v_max].
        """
        super(RainbowNetwork, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        
        self.value_noisy = NoisyLinear(n_neurons, self.n_atoms)
        self.value_stream = nn.Sequential(self.value_noisy)
        
        self.advantage_noisy = NoisyLinear(n_neurons, n_actions*n_atoms)
        self.advantage_stream = nn.Sequential(self.advantage_noisy)
    
    def forward(self, x):
        distribution = self.distribution(x)
        return torch.sum(self.support*distribution, dim=2)
    
    def logits(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        
        advantage = self.advantage_stream(x)
        
        value = value.view(-1, 1, self.n_atoms)
        advantage = advantage.view(-1, self.n_actions, self.n_atoms)
        
        return value + (advantage - advantage.mean(1, keepdim=True))
    
    def distribution(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=2)

    def reset_noise(self):
        self.value_noisy.reset_noise()
        self.advantage_noisy.reset_noise()

class RainbowAtariNetwork(nn.Module):
    """Rainbow network architecture suited for Atari 2600 environment.
    
    Specifically, it is a dueling, noisy, distributional architecture."""
    
    def __init__(self, n_actions, n_atoms=51, v_min=-10, v_max=10):
        """Creates the layers.
        
        Args:
            n_actions: int, number of actions possible for the agent.
            n_atoms: int, number of atoms for discretization of the support.
        """
        super(RainbowAtariNetwork, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.value_noisy1 = NoisyLinear(7 * 7 * 64, 512)
        self.value_noisy2 = NoisyLinear(512, self.n_atoms)
        self.value_stream = nn.Sequential(
            self.value_noisy1,
            nn.ReLU(),
            self.value_noisy2)
        
        self.advantage_noisy1 = NoisyLinear(7 * 7 * 64, 512)
        self.advantage_noisy2 = NoisyLinear(512, self.n_actions*self.n_atoms)
        self.advantage_stream = nn.Sequential(
            self.advantage_noisy1,
            nn.ReLU(),
            self.advantage_noisy2)
    
    def forward(self, x):
        distribution = self.distribution(x)
        return torch.sum(self.support*distribution, dim=2)
    
    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        value = self.value_stream(x)
        
        advantage = self.advantage_stream(x)
        
        value = value.view(-1, 1, self.n_atoms)
        advantage = advantage.view(-1, self.n_actions, self.n_atoms)
        
        return value + (advantage - advantage.mean(1, keepdim=True))
    
    def distribution(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=2)
    
    def reset_noise(self):
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.advantage_noisy1.reset_noise()
        self.advantage_noisy2.reset_noise()
