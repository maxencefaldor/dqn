#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a Rainbow agent."""

import torch
import torch.nn as nn
import torch.optim as optim

from agents.ddqn_agent import DDQNAgent


class RainbowAgent(DDQNAgent):
    """Implementation of a Rainbow agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 gamma=0.99,
                 n=1,
                 n_gradient_steps=1,
                 beta=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 batch_size=32,
                 per=True,
                 buffer_size=1e6):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take.
            network: `torch.nn`, neural network used to approximate Q.
            lr: float, learning rate.
            gamma: float, discount rate.
            n: int, number of steps of bootstrapping.
            n_gradient_steps: int, number of gradient steps taken during a
                time step.
            beta: float, update period for the target network if beta
                is a positive integer. Soft update parameter for the target
                network if beta is a float in (0, 1).
            epsilon_min: float, the minimum epsilon value during training.
            epsilon_decay: int, epsilon decay parameter.
            batch_size: int, batch size.
            per: bool, If True, use prioritized experience replay, else use
                uniformly sampled experience replay.
            buffer_size: int, capacity of the replay buffer.
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
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=lr, epsilon=0.0003125)
