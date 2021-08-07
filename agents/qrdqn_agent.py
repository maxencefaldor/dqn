#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a QR-DQN agent.

Implementation of the DQN algorithm and six independent improvements as
described in "Rainbow: Combining Improvements in Deep Reinforcement Learning",
Hessel et al. (2017), except that the distributional algorithm is Quantile
Regression Q-Learning.

Specifically, the 6 improvements are:
    * double DQN
    * prioritized experience replay
    * dueling network architecture
    * multi-step bootstrapping
    * distributional RL (QR)
    * noisy networks
"""

from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.ddqn_agent import DDQNAgent


class QRDQNAgent(DDQNAgent):
    """Implementation of a QR-DQN agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 n_atoms=51,
                 v_min=-10.,
                 v_max=10.,
                 gamma=0.99,
                 n=1,
                 n_gradient_steps=1,
                 beta=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 batch_size=32,
                 buffer_size=1e6,
                 per=True):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take at any state.
            network: `torch.nn`, neural network used to approximate the
                Q-value.
            lr: float, learning rate.
            criterion: `nn.modules.loss`, loss used to train the network.
            n_atoms: int, the number of bins of the value distribution.
            v_min: float, the value distribution support is [v_min, v_max].
            v_max: float, the value distribution support is [v_min, v_max].
            gamma: float, discount rate.
            n: int, number of steps of bootstrapping.
            n_gradient_steps: int, number of gradient steps taken by time step.
            beta: float, update period for the target network if beta
                is a positive integer. Soft update parameter for the target
                network if beta is a float in (0, 1).
            epsilon_min: float, the minimum epsilon value during training.
            epsilon_decay: int, epsilon decay parameter.
            batch_size: int, batch size.
            buffer_size: int, capacity of the replay buffer.
            per: bool, If True, use prioritized experience replay, else use
                uniformly sampled experience replay.
        """
        DDQNAgent.__init__(self,
                           device=device,
                           n_actions=n_actions,
                           network=network,
                           lr=lr,
                           gamma=gamma,
                           n=n,
                           n_gradient_steps=n_gradient_steps,
                           beta=beta,
                           epsilon_min=epsilon_min,
                           epsilon_decay=epsilon_decay,
                           batch_size=batch_size,
                           per=per,
                           buffer_size=buffer_size)
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, n_atoms, device=self._device)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=lr, eps=0.0003125)
        self.criterion = self._cross_entropy_with_logits
    
    def _cross_entropy_with_logits(self, labels, logits):
        return -torch.sum(labels * F.log_softmax(logits, dim=1),
                          dim=1).unsqueeze(1)