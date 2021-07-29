#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.dqn_agent import D2QNAgent


class D3QNAgent(D2QNAgent):
    """Implementation of the Dueling Double DQN agent"""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 gamma=0.99,
                 n=1,
                 batch_size=32,
                 n_gradient_steps=1,
                 epsilon_min=0.01,
                 epsilon_decay=2000,
                 buffer_size=1e6,
                 beta=1):
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
        D2QNAgent.__init__(self,
                           device=device,
                           n_actions=n_actions,
                           network=network,
                           lr=lr,
                           gamma=gamma,
                           n=n,
                           batch_size=batch_size,
                           n_gradient_steps=n_gradient_steps,
                           epsilon_min=epsilon_min,
                           epsilon_decay=epsilon_decay,
                           buffer_size=buffer_size,
                           beta=beta)