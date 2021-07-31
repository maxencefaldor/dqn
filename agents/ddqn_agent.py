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
    
    def _next_state_q_values(self, next_states):
        """Returns the next state Q-values.
        
        Args:
            next_state_batch: `torch.Tensor`, batch of next state.
        
        Returns:
            `torch.Tensor`, next state Q-values.
        """
        return self.target_network(next_states).gather(1, self.network(next_states).max(1)[1].unsqueeze(1)).detach()
