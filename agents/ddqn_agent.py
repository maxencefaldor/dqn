#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a Double DQN agent."""

import torch

from agents.dqn_agent import DQNAgent


class DDQNAgent(DQNAgent):
    """Implementation of a Double DQN agent."""
    
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
                 per=False,
                 buffer_size=1e6):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take.
            network: `torch.nn`, neural network used to approximate Q.
            lr: float, learning rate.
            gamma: float, discount rate.
            batch_size: int, batch size.
            n_gradient_steps: int, number of gradient steps taken during a
                time step.
            epsilon_min: float, the minimum epsilon value during training.
            epsilon_decay: int, epsilon decay parameter.
            buffer_size: int, capacity of the replay buffer.
            beta: float, update period for the target network if beta
                is a positive integer. Soft update parameter for the target
                network if beta is a float in (0, 1).
        """
        DQNAgent.__init__(self,
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
    
    def _next_state_q(self, next_state_batch):
        """Returns the next_state Q-values
        
        Args:
            next_state_batch: tuple, batch of next state.
        
        Returns:
            torch.Tensor, Q-values of the batch.
        """
        return self.target_network(next_state_batch).gather(1, self.network(next_state_batch).max(1)[1].unsqueeze(1)).detach()
