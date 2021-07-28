#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from agent.dqn_agent import DQNAgent


class D2QNAgent(DQNAgent):
    """Implementation of the Double DQN agent"""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 gamma=0.99,
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
        
        DQNAgent.__init__(self,
                          device=device,
                          n_actions=n_actions,
                          network=network,
                          lr=lr,
                          gamma=gamma,
                          batch_size=batch_size,
                          n_gradient_steps=n_gradient_steps,
                          epsilon_min=epsilon_min,
                          epsilon_decay=epsilon_decay,
                          buffer_size=buffer_size,
                          beta=beta)
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        batch = [*zip(*batch)]
        
        state_batch = torch.stack(batch[0]).to(self._device)
        action_batch = torch.stack(batch[1]).to(self._device)
        reward_batch = torch.stack(batch[2]).to(self._device)
        next_state_batch = torch.stack(batch[3]).to(self._device)
        done_batch = torch.stack(batch[4]).to(self._device)
        
        state_action_values = self.network(state_batch).gather(1, action_batch)
        next_state_action_values = self.target_network(next_state_batch).gather(1, self.network(next_state_batch).max(1)[1].unsqueeze(1)).detach()
        expected_state_action_values = reward_batch + self.gamma * next_state_action_values * (1 - done_batch)
        
        loss = self.criterion(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_network()
