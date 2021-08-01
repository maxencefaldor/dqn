#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of a Rainbow agent."""

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
                 criterion=nn.CrossEntropyLoss,
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
                 per=True,
                 buffer_size=1e6):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take.
            network: `torch.nn`, neural network used to approximate Q.
            lr: float, learning rate.
            criterion: `nn.modules.loss`, loss used to train the network.
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
                           criterion=criterion,
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
    
    def greedy_action(self, state):
        """Returns an action following a greedy policy.
        
        Args:
            state: torch.Tensor, state of the agent.
        
        Returns:
            int, greedy action.
        """
        with torch.no_grad():
            _, _, q_values = self.network(torch.Tensor(state).to(self._device).unsqueeze(0))
            return torch.argmax(q_values).item()
    
    def _target_state_q_values(self, rewards, next_states, dones):
        tiled_support = self.support.tile((self.batch_size, 1))
        target_support = rewards + self.gamma_n * tiled_support * (1 - dones)
        
        logits, probabilities, q_values = self.target_network(next_states)
        max_actions = q_values.max(1)[1]
        next_probabilities = torch.stack(
            [probabilities[i, max_actions[i], :] for i in range(self.batch_size)])
        
        return self._project_distribution(
            target_support, next_probabilities).detach()
    
    def _project_distribution(self, supports, weights):
        target_support_deltas = self.support[1:] - self.support[:-1]
        delta_z = target_support_deltas[0]
        
        clipped_support = torch.clamp(supports, min=self.v_min,
                                      max=self.v_max)[:, None, :]
        tiled_support = clipped_support.expand(-1, self.n_atoms, -1)
        
        reshaped_target_support = self.support.tile((self.batch_size, 1)).view(
            self.batch_size, self.n_atoms, -1)
        numerator = torch.abs(tiled_support - reshaped_target_support)
        clipped_quotient = torch.clamp(1 - (numerator/delta_z), min=0, max=1)
        
        weights = weights[:, None, :]
        inner_prod = clipped_quotient * weights
        return torch.sum(inner_prod, 2)

    def learn(self):
        """Learns the Q-value from the replay memory."""
        if len(self.replay_buffer) - self.n + 1 < self.batch_size:
            return
        
        self.replay_buffer.sample(self.batch_size)
        
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        dones = self.replay_buffer.dones
        
        logits, _, _ = self.network(states)
        state_q_values = torch.stack(
            [logits[i, actions.view(-1)[i], :] for i in range(self.batch_size)])
        with torch.no_grad():
            target_state_q_values = self._target_state_q_values(rewards,
                                                                next_states,
                                                                dones)
        
        if self.per:
            loss = -torch.sum(target_state_q_values * F.log_softmax(state_q_values, dim=1),
                              dim=1).unsqueeze(1)
            
            errors = loss
            for i, index in enumerate(self.replay_buffer.indices):
                self.replay_buffer.update(index, errors[i][0].item())

            loss *= self.replay_buffer.is_weight
            loss = loss.mean()
        else:
            loss = -torch.sum(target_state_q_values * F.log_softmax(state_q_values, dim=1),
                              dim=1).unsqueeze(1)
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target_network()
