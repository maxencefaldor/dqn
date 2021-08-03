#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a Double DQN agent.

Specifically, the algorithm implements "Deep Reinforcement Learning with Double
Q-learning", Hasselt et al. (2015).

In addition to this, the agent can perform
    * prioritized experience replay
    * multi-step bootstrapping
"""

import torch.nn as nn

from agents.dqn_agent import DQNAgent


class DDQNAgent(DQNAgent):
    """Implementation of a Double DQN agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 criterion=nn.MSELoss,
                 gamma=0.99,
                 n=1,
                 n_gradient_steps=1,
                 beta=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 batch_size=32,
                 buffer_size=1e6,
                 per=False):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take at any state.
            network: `torch.nn`, neural network used to approximate the
                Q-value.
            lr: float, learning rate.
            criterion: `nn.modules.loss`, loss used to train the network.
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
        DQNAgent.__init__(self,
                          device=device,
                          n_actions=n_actions,
                          network=network,
                          lr=lr,
                          criterion=criterion,
                          gamma=gamma,
                          n=n,
                          n_gradient_steps=n_gradient_steps,
                          beta=beta,
                          epsilon_min=epsilon_min,
                          epsilon_decay=epsilon_decay,
                          batch_size=batch_size,
                          buffer_size=buffer_size,
                          per=per)
    
    def _next_state_q_values(self, next_states):
        """Returns the next state Q-values.
        
        Args:
            next_state_batch: `torch.Tensor`, batch of next state.
        
        Returns:
            `torch.Tensor`, next state Q-values.
        """
        return self.target_network(next_states).gather(1, self.network(next_states).max(1)[1].unsqueeze(1)).detach()
