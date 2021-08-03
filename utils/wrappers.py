#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym.wrappers import TimeLimit
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from gym import logger
logger.set_level(gym.logger.DISABLED)

import numpy as np


class CartPoleSwingUp(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.theta_dot_threshold = 4*np.pi

    def reset(self):
        self.env.env.state = [0, 0, np.pi, 0] + super().reset()
        self.env.env.steps_beyond_done = None
        return np.array(self.env.env.state)

    def step(self, action):
        state, reward, done, _ = super().step(action)
        x, x_dot, theta, theta_dot = state
        
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta_dot < -self.theta_dot_threshold \
               or theta_dot > self.theta_dot_threshold
        
        if done:
            # game over
            reward = -10.
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        else:
            if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.

        return np.array(self.state), reward, done, {}

def make_cartpole_swing_up(env_id, max_episode_steps):
    """Returns a CartPole Swing Up environment
    
    The environment is similar to the classic control CartPole environment,
    though more complex. The pole starts hanging down and the cart must first
    swing the pole to an upright position before balancing it as in normal
    CartPole.
    
    Args:
        env_id: str, environment id.
        max_episode_steps: int, maximum number of steps by episode.
    
    Returns:
        Gym environment.
    """
    env = gym.make(env_id)
    assert "CartPole" in env.spec.id
    env = CartPoleSwingUp(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

class AtariWrapper(AtariPreprocessing):
    def __init__(self, env, action_map, **kwargs):
        super(AtariWrapper, self).__init__(env, **kwargs)
        self.action_map = action_map
    
    def step(self, action):
        return super(AtariWrapper, self).step(self.action_map[action])

def make_atari(env_id, action_map):
    """Returns an Atari 2600 Gym environment.
    
    The environment follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
    Problems for General Agents".
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on
        reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not
        recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional
    * Frame Stacking
    
    Args:
        env_id: str, environment id.
        action_map: dict, a dictionnary to map integer from [0, ..., n] to a
            different list of integer [a_1, ... a_n]. For example, in the Pong
            environment, the only two useful actions are 4 and 5. A dict can
            be used to map these action back to 0 and 1.
    
    Returns:
        Gym environment.
    """
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = AtariWrapper(env,
                       action_map=action_map,
                       noop_max=0,
                       frame_skip=4,
                       terminal_on_life_loss=True,
                       grayscale_obs=True,
                       scale_obs=True)
    env = FrameStack(env, 4)
    return env
