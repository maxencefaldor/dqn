#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a Rainbow agent.

Implementation of the DQN algorithm and six independent improvements as
described in "Rainbow: Combining Improvements in Deep Reinforcement Learning",
Hessel et al. (2017).

Specifically, the 6 improvements are:
    * double DQN
    * prioritized experience replay
    * dueling network architecture
    * multi-step bootstrapping
    * distributional RL (C51)
    * noisy networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.ddqn_agent import DDQNAgent


class RainbowAgent(DDQNAgent):
    """Implementation of a Rainbow agent."""
    
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
    
    def _target_state_q_values(self, rewards, next_states, dones):
        tiled_support = self.support.tile((self.batch_size, 1))
        target_support = rewards + self.gamma_n * tiled_support * (1 - dones)
        
        distribution = self.target_network.distribution(next_states)
        max_actions = self.network(next_states).max(1)[1].unsqueeze(1)\
            .unsqueeze(1).expand(self.batch_size, 1, self.n_atoms)
        next_distribution = distribution.gather(1, max_actions).squeeze(1)
        
        return self._project_distribution(
            target_support, next_distribution).detach()
    
    def _project_distribution(self, supports, distributions):
        """Projects a batch of supports and distributions onto the base
        support.
        
        This function implements equation 7 in "A Distributional Perspective on
        Reinforcement Learning", Bellemare et al. (2017).
        
        Args:
            supports: `torch.Tensor`, supports for the distributions.
            distributions: `torch.Tensor`, probability distributions to project
                on the base support [-v_min, v_min].
        
        Returns: `torch.Tensor`, batch of projected distributions.
        """
        target_support_deltas = self.support[1:] - self.support[:-1]
        delta_z = target_support_deltas[0]
        
        clipped_support = torch.clamp(supports, min=self.v_min,
                                      max=self.v_max)[:, None, :]
        tiled_support = clipped_support.expand(-1, self.n_atoms, -1)
        
        reshaped_target_support = self.support.tile((self.batch_size, 1)).view(
            self.batch_size, self.n_atoms, -1)
        numerator = torch.abs(tiled_support - reshaped_target_support)
        clipped_quotient = torch.clamp(1 - (numerator/delta_z), min=0, max=1)
        
        distributions = distributions[:, None, :]
        inner_prod = clipped_quotient * distributions
        return torch.sum(inner_prod, 2)

    def learn(self):
        """Learns the Q-value from experience replay."""
        if len(self.replay_buffer) - self.n + 1 < self.batch_size:
            return
        
        self.replay_buffer.sample(self.batch_size)
        
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        dones = self.replay_buffer.dones
        
        logits = self.network.logits(states)
        state_q_values = logits.gather(1, actions.unsqueeze(1).expand(
            self.batch_size, 1, self.n_atoms)).squeeze(1)
        with torch.no_grad():
            target_state_q_values = self._target_state_q_values(rewards,
                                                                next_states,
                                                                dones)
        
        if self.per:
            loss = self.criterion(target_state_q_values, state_q_values)
            
            errors = loss
            for i, index in enumerate(self.replay_buffer.indices):
                self.replay_buffer.update(index, errors[i].item())

            loss *= self.replay_buffer.is_weight
            loss = loss.mean()
        else:
            loss = self.criterion(target_state_q_values, state_q_values)
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_network()
